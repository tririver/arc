from __future__ import annotations

from typing import Iterable


REGENERATABLE_LANES = (
    "segmentation", "glossary", "guide", "translation", "commentary", "review"
)


class RegenerationRequestError(ValueError):
    pass


def normalize_regeneration_lanes(
    values: Iterable[str], *, confirm_expensive_all: bool = False
) -> tuple[str, ...]:
    requested = [str(value).strip().casefold() for value in values if str(value).strip()]
    unknown = set(requested).difference((*REGENERATABLE_LANES, "all"))
    if unknown:
        raise RegenerationRequestError(
            f"unsupported regeneration lane: {', '.join(sorted(unknown))}"
        )
    if "all" in requested:
        if len(set(requested)) != 1:
            raise RegenerationRequestError("all cannot be combined with scoped lanes")
        if not confirm_expensive_all:
            raise RegenerationRequestError(
                "--regenerate all requires --confirm-expensive-regeneration"
            )
        return REGENERATABLE_LANES
    return tuple(lane for lane in REGENERATABLE_LANES if lane in set(requested))


def reject_broad_force(force: bool, lanes: Iterable[str]) -> None:
    if force:
        raise RegenerationRequestError(
            "--force no longer regenerates companion content; use --regenerate LANE"
        )
