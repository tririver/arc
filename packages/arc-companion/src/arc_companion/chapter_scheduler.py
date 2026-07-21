from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import BoundedSemaphore, Event, Lock
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
    first_failure: list[BaseException] = []
    failure_lock = Lock()

    def remember_failure(exc: BaseException) -> None:
        with failure_lock:
            if not first_failure:
                first_failure.append(exc)
        stopped.set()

    def guarded(call: Callable[..., Any], *args: Any) -> Any:
        if stopped.is_set():
            raise _ChapterPipelineStopped("chapter pipeline stopped before another call")
        with budget:
            if stopped.is_set():
                raise _ChapterPipelineStopped("chapter pipeline stopped before another call")
            try:
                return call(*args)
            except BaseException as exc:
                remember_failure(exc)
                raise

    def prepare(chapter: dict[str, Any]) -> PreparedChapter:
        with ThreadPoolExecutor(max_workers=2) as phase:
            guide_future = phase.submit(guarded, prepare_guide, chapter)
            segments_future = phase.submit(guarded, prepare_segments, chapter)
            try:
                guide = guide_future.result()
                segments = segments_future.result()
            except BaseException:
                if first_failure:
                    raise first_failure[0]
                raise
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
            for name, future in futures.items():
                try:
                    outputs[name] = future.result()
                except BaseException:
                    pass
            if first_failure:
                raise first_failure[0]
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
        for chapter_id, future in pending:
            if not chapter_id:
                raise ValueError("chapter is missing chapter_id")
            results[chapter_id] = future.result()
    return results


class _ChapterPipelineStopped(RuntimeError):
    """Internal signal used to drain already queued work without new calls."""
