from __future__ import annotations

from collections.abc import Iterable
import re


ARXIV_ARCHIVE_PATTERN = (
    r"(?:astro-ph|cond-mat|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|math-ph|"
    r"nlin|nucl-ex|nucl-th|physics|quant-ph|q-bio|q-fin|stat|math|cs|econ|eess)"
)
NEW_STYLE_ARXIV_RE = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
OLD_STYLE_ARXIV_RE = re.compile(
    rf"^(?:arxiv:)?({ARXIV_ARCHIVE_PATTERN}/\d{{7}})(?:v\d+)?$",
    re.IGNORECASE,
)
ARXIV_URL_RE = re.compile(
    rf"\b(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/"
    rf"((?:\d{{4}}\.\d{{4,5}})|(?:{ARXIV_ARCHIVE_PATTERN}/\d{{7}}))(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)
ARXIV_URL_EXTRACT_RE = re.compile(
    rf"\b(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/"
    rf"((?:\d{{4}}\.\d{{4,5}})|(?:{ARXIV_ARCHIVE_PATTERN}/\d{{7}}))(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)
ARXIV_PREFIX_EXTRACT_RE = re.compile(
    rf"\barxiv\s*:\s*((?:\d{{4}}\.\d{{4,5}})|(?:{ARXIV_ARCHIVE_PATTERN}/\d{{7}}))(?:v\d+)?",
    re.IGNORECASE,
)
BARE_NEW_STYLE_ARXIV_RE = re.compile(r"(?<![\w./-])(\d{4}\.\d{4,5})(?:v\d+)?(?![\w/-])", re.IGNORECASE)
BARE_OLD_STYLE_ARXIV_RE = re.compile(
    rf"(?<![\w/-])({ARXIV_ARCHIVE_PATTERN}/\d{{7}})(?:v\d+)?(?![\w/-])",
    re.IGNORECASE,
)
INSPIRE_RECID_RE = re.compile(r"^(?:inspire:|recid:)(\d+)$", re.IGNORECASE)
INSPIRE_EXTRACT_RE = re.compile(r"\b(?:inspire|recid)\s*:\s*(\d+)\b", re.IGNORECASE)
INSPIRE_URL_EXTRACT_RE = re.compile(
    r"\b(?:https?://)?(?:www\.)?inspirehep\.net/(?:api/)?literature/(\d+)\b",
    re.IGNORECASE,
)
DOI_ID_RE = re.compile(
    r"^(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/|(?:dx\.)?doi\.org/)?"
    r"(10\.\d{4,9}/[^\s<>\"?#]+)$",
    re.IGNORECASE,
)
DOI_EXTRACT_RE = re.compile(
    r"\b(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/|(?:dx\.)?doi\.org/)?"
    r"(10\.\d{4,9}/[^\s<>\"?#]+)",
    re.IGNORECASE,
)


def normalize_paper_id(identifier: str) -> str:
    text = (identifier or "").strip()
    if not text:
        return ""

    if arxiv_id := _normalized_arxiv_id(text):
        return f"arXiv:{arxiv_id}"
    if match := INSPIRE_RECID_RE.match(text):
        return f"inspire:{match.group(1)}"
    if match := DOI_ID_RE.match(text):
        return f"doi:{_normalize_doi(match.group(1))}"
    return text


def arxiv_path_id(identifier: str) -> str:
    return _normalized_arxiv_id(identifier)


def inspire_recid(identifier: str) -> str:
    normalized = normalize_paper_id(identifier)
    if not normalized.startswith("inspire:"):
        return ""
    return normalized.split(":", 1)[1]


def doi_value(identifier: str) -> str:
    normalized = normalize_paper_id(identifier)
    if not normalized.startswith("doi:"):
        return ""
    return normalized.split(":", 1)[1]


def paper_ids_safe_dir_name(identifiers: Iterable[str] | str) -> str:
    values = [identifiers] if isinstance(identifiers, str) else identifiers
    normalized_ids = _dedupe_ids([normalize_paper_id(str(identifier)) for identifier in values if str(identifier)])
    return "_x_".join(_safe_dir_component(identifier) for identifier in normalized_ids)


def extract_paper_ids(text: str) -> list[str]:
    """Extract normalized paper identifiers from natural-language text."""
    source = text or ""
    found: list[tuple[int, int, str]] = []

    def add(match: re.Match[str], identifier: str) -> None:
        if identifier:
            found.append((match.start(), match.end(), identifier))

    for match in DOI_EXTRACT_RE.finditer(source):
        add(match, f"doi:{_normalize_doi(match.group(1))}")
    for match in ARXIV_URL_EXTRACT_RE.finditer(source):
        if arxiv_id := _normalized_arxiv_id(f"arXiv:{match.group(1)}"):
            add(match, f"arXiv:{arxiv_id}")
    for match in ARXIV_PREFIX_EXTRACT_RE.finditer(source):
        if arxiv_id := _normalized_arxiv_id(f"arXiv:{match.group(1)}"):
            add(match, f"arXiv:{arxiv_id}")
    for match in INSPIRE_URL_EXTRACT_RE.finditer(source):
        add(match, f"inspire:{match.group(1)}")
    for match in INSPIRE_EXTRACT_RE.finditer(source):
        add(match, f"inspire:{match.group(1)}")

    clear = _dedupe_matches(found)
    remaining = _blank_spans(source, [(start, end) for start, end, _ in clear])

    ambiguous: list[tuple[int, int, str]] = []
    for match in BARE_NEW_STYLE_ARXIV_RE.finditer(remaining):
        if arxiv_id := _normalized_arxiv_id(match.group(1)):
            ambiguous.append((match.start(), match.end(), f"arXiv:{arxiv_id}"))
    for match in BARE_OLD_STYLE_ARXIV_RE.finditer(remaining):
        if arxiv_id := _normalized_arxiv_id(match.group(1).lower()):
            ambiguous.append((match.start(), match.end(), f"arXiv:{arxiv_id}"))

    return _dedupe_ids([identifier for _, _, identifier in sorted(clear + _dedupe_matches(ambiguous))])


def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip(), flags=re.IGNORECASE)


def _normalized_arxiv_id(identifier: str) -> str:
    text = (identifier or "").strip()
    if not text:
        return ""
    if match := ARXIV_URL_RE.search(text):
        return _canonical_arxiv_id(match.group(1))
    if match := NEW_STYLE_ARXIV_RE.match(text):
        return match.group(1) if _valid_new_style_arxiv_id(match.group(1)) else ""
    if match := OLD_STYLE_ARXIV_RE.match(text):
        return _canonical_arxiv_id(match.group(1))
    return ""


def _canonical_arxiv_id(arxiv_id: str) -> str:
    if "/" not in arxiv_id:
        return arxiv_id if _valid_new_style_arxiv_id(arxiv_id) else ""
    archive, number = arxiv_id.split("/", 1)
    return f"{archive.lower()}/{number}"


def _valid_new_style_arxiv_id(arxiv_id: str) -> bool:
    match = re.fullmatch(r"(\d{2})(\d{2})\.\d{4,5}", _strip_version(arxiv_id))
    if not match:
        return False
    year = int(match.group(1))
    month = int(match.group(2))
    if month < 1 or month > 12:
        return False
    if year < 7:
        return False
    if year == 7 and month < 4:
        return False
    return True


def _normalize_doi(value: str) -> str:
    doi = _strip_identifier_punctuation(value)
    return doi.lower()


def _strip_identifier_punctuation(value: str) -> str:
    text = (value or "").strip()
    text = text.split("?", 1)[0].split("#", 1)[0]
    while text and text[-1] in ".,;:":
        text = text[:-1]
    pairs = {")": "(", "]": "[", "}": "{"}
    while text and text[-1] in pairs and text.count(text[-1]) > text.count(pairs[text[-1]]):
        text = text[:-1]
    return text


def _dedupe_matches(matches: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    selected: list[tuple[int, int, str]] = []
    for match in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        start, end, _ = match
        if any(start < selected_end and end > selected_start for selected_start, selected_end, _ in selected):
            continue
        selected.append(match)
    return sorted(selected, key=lambda item: item[0])


def _blank_spans(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for index in range(start, min(end, len(chars))):
            chars[index] = " "
    return "".join(chars)


def _dedupe_ids(identifiers: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for identifier in identifiers:
        key = identifier.lower()
        if key not in seen:
            seen.add(key)
            out.append(identifier)
    return out


def _safe_dir_component(identifier: str) -> str:
    normalized = normalize_paper_id(identifier)
    if arxiv_id := arxiv_path_id(normalized):
        source = arxiv_id
    elif normalized.startswith("doi:"):
        source = normalized
    elif normalized.startswith("inspire:"):
        source = normalized
    else:
        source = normalized
    safe = re.sub(r"[^A-Za-z0-9.-]+", "_", source.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "paper"
