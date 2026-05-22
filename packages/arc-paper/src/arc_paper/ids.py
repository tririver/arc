from __future__ import annotations

import re


NEW_STYLE_ARXIV_RE = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
OLD_STYLE_ARXIV_RE = re.compile(
    r"^(?:arxiv:)?("
    r"(?:astro-ph|cond-mat|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|math-ph|"
    r"nlin|nucl-ex|nucl-th|physics|quant-ph|q-bio|q-fin|stat|math|cs|econ|eess)"
    r"/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)
ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/((?:\d{4}\.\d{4,5})|(?:[a-z-]+/\d{7}))(?:v\d+)?",
    re.IGNORECASE,
)
INSPIRE_RECID_RE = re.compile(r"^(?:inspire:|recid:)(\d+)$", re.IGNORECASE)


def normalize_paper_id(identifier: str) -> str:
    text = (identifier or "").strip()
    if not text:
        return ""

    if match := ARXIV_URL_RE.search(text):
        return f"arXiv:{match.group(1)}"
    if match := NEW_STYLE_ARXIV_RE.match(text):
        return f"arXiv:{match.group(1)}"
    if match := OLD_STYLE_ARXIV_RE.match(text):
        return f"arXiv:{match.group(1)}"
    if text.lower().startswith("arxiv:"):
        return "arXiv:" + _strip_version(text.split(":", 1)[1])
    if match := INSPIRE_RECID_RE.match(text):
        return f"inspire:{match.group(1)}"
    return text


def arxiv_path_id(identifier: str) -> str:
    normalized = normalize_paper_id(identifier)
    if not normalized.startswith("arXiv:"):
        return ""
    return normalized.split(":", 1)[1]


def inspire_recid(identifier: str) -> str:
    normalized = normalize_paper_id(identifier)
    if not normalized.startswith("inspire:"):
        return ""
    return normalized.split(":", 1)[1]


def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip(), flags=re.IGNORECASE)
