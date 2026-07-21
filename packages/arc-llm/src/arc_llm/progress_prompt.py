from __future__ import annotations


RUNTIME_PROGRESS_CONTRACT_VERSION = "1"
RUNTIME_PROGRESS_CONTRACT_MARKER = (
    f'<arc_llm_runtime_progress_contract version="{RUNTIME_PROGRESS_CONTRACT_VERSION}">'
)
RUNTIME_PROGRESS_SESSION_MARKER = (
    f'<arc_llm_runtime_progress_contract version="{RUNTIME_PROGRESS_CONTRACT_VERSION}" scope="session" />'
)


def runtime_progress_contract() -> str:
    """Return the public, provider-portable ARC worker progress contract."""

    return "\n".join(
        [
            RUNTIME_PROGRESS_CONTRACT_MARKER,
            "## ARC-LLM Runtime Progress Contract v1",
            "For work with multiple stages or long-running steps, report progress as soon as a meaningful milestone is reached; do not wait until all work is complete.",
            "Send progress only through the runtime's out-of-band progress, commentary, or tool-event channel. Never put progress messages in the final answer, final JSON object, or any field of the requested output schema.",
            "A useful milestone states: what completed; a concrete result or evidence; the path of any reusable artifact or checkpoint when available; what happens next; and any blocker. Report before starting another potentially long stage so completed work can be inspected and resumed.",
            "Do not emit empty heartbeats such as 'still alive', repeated plans, or status messages without a new result. Short single-stage calls do not need artificial progress updates.",
            "Do not expose private chain-of-thought. Report concise outcomes, evidence, decisions, tool status, and artifact locations only.",
            "If the final output is strict structured data, keep it schema-valid and use only the out-of-band channel for progress.",
            "</arc_llm_runtime_progress_contract>",
        ]
    )


def has_runtime_progress_contract(prompt: str) -> bool:
    """Return whether *prompt* already contains the current contract."""

    return RUNTIME_PROGRESS_CONTRACT_MARKER in prompt


def ensure_runtime_progress_contract(prompt: str) -> str:
    """Append the current contract once, preserving the caller's prompt text."""

    text = str(prompt)
    if has_runtime_progress_contract(text):
        return text
    return text.rstrip() + "\n\n" + runtime_progress_contract() + "\n"


def ensure_runtime_progress_marker(prompt: str) -> str:
    """Append the compact marker used after a session-generation bootstrap."""

    text = str(prompt)
    if has_runtime_progress_contract(text) or RUNTIME_PROGRESS_SESSION_MARKER in text:
        return text
    return text.rstrip() + "\n\n" + RUNTIME_PROGRESS_SESSION_MARKER + "\n"


def apply_runtime_progress_contract(
    prompt: str,
    *,
    scope: str,
    generation_bootstrap: bool = True,
) -> str:
    if scope not in {"call", "session"}:
        raise ValueError("progress_contract_scope must be call or session")
    if scope == "call" or generation_bootstrap:
        return ensure_runtime_progress_contract(prompt)
    return ensure_runtime_progress_marker(prompt)
