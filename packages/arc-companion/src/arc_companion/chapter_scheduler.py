from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import BoundedSemaphore, Event
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class PreparedChapter:
    chapter: dict[str, Any]
    guide: dict[str, Any]
    segments: list[dict[str, Any]]


def run_chapter_pipeline(
    chapters: list[Mapping[str, Any]],
    *,
    workers: int,
    prepare_guide: Callable[[Mapping[str, Any]], dict[str, Any]],
    prepare_segments: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    run_translation: Callable[[PreparedChapter, Mapping[str, Any]], Any] | None,
    run_companion: Callable[[PreparedChapter, Mapping[str, Any]], Any],
    stop_after_first_chapter: bool = False,
    stop_event: Event | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run chapters concurrently while each lane advances in source order.

    Every callback that may invoke an LLM is guarded by the same global
    semaphore.  Lane callbacks receive one segment at a time, so a caller can
    persist its accepted ledger before this scheduler submits the successor.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    selected = [dict(chapters[0])] if stop_after_first_chapter and chapters else [dict(item) for item in chapters]
    budget = BoundedSemaphore(workers)
    stopped = stop_event or Event()
    def is_stopped() -> bool:
        return stopped.is_set() or bool(cancel_check is not None and cancel_check())

    def guarded(call: Callable[..., Any], *args: Any) -> Any:
        if is_stopped():
            raise _ChapterPipelineStopped("chapter pipeline stopped before another call")
        with budget:
            if is_stopped():
                raise _ChapterPipelineStopped("chapter pipeline stopped before another call")
            return call(*args)

    def prepare(chapter: dict[str, Any]) -> PreparedChapter:
        with ThreadPoolExecutor(max_workers=2) as phase:
            guide_future = phase.submit(guarded, prepare_guide, chapter)
            segments_future = phase.submit(guarded, prepare_segments, chapter)
            failures: list[BaseException] = []
            guide = None
            segments = None
            for future, target in ((guide_future, "guide"), (segments_future, "segments")):
                try:
                    value = future.result()
                except BaseException as exc:
                    failures.append(exc)
                else:
                    if target == "guide":
                        guide = value
                    else:
                        segments = value
            if failures:
                raise failures[0]
            assert isinstance(guide, dict) and isinstance(segments, list)
            return PreparedChapter(chapter, guide, segments)

    def lane(prepared: PreparedChapter, call: Callable[[PreparedChapter, Mapping[str, Any]], Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for segment in prepared.segments:
            segment_id = str(segment.get("segment_id") or "")
            if not segment_id:
                raise ValueError("prepared segment is missing segment_id")
            output[segment_id] = guarded(call, prepared, segment)
        return output

    def run_chapter(chapter: dict[str, Any]) -> dict[str, Any]:
        prepared = prepare(chapter)
        with ThreadPoolExecutor(max_workers=2 if run_translation is not None else 1) as phase:
            futures = {}
            if run_translation is not None:
                futures["translation"] = phase.submit(lane, prepared, run_translation)
            futures["companion"] = phase.submit(lane, prepared, run_companion)
            outputs: dict[str, Any] = {}
            failures: list[BaseException] = []
            for name, future in futures.items():
                try:
                    outputs[name] = future.result()
                except BaseException as exc:
                    failures.append(exc)
            if failures:
                raise failures[0]
            return {
                "guide": prepared.guide,
                "segments": prepared.segments,
                "translation": {},
                **outputs,
            }

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(max(1, len(selected)), workers)) as executor:
        pending: list[tuple[str, Future[dict[str, Any]]]] = [
            (str(chapter.get("chapter_id") or ""), executor.submit(run_chapter, chapter))
            for chapter in selected
        ]
        failures: list[BaseException] = []
        for chapter_id, future in pending:
            if not chapter_id:
                raise ValueError("chapter is missing chapter_id")
            try:
                results[chapter_id] = future.result()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
    return results


class _ChapterPipelineStopped(RuntimeError):
    """Internal signal used to drain already queued work without new calls."""
