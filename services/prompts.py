"""
Agent prompts, loaded from prompts/<version>/.

Each version folder holds the whole set the pipeline uses (settings.prompt_version). A version
name alone does not pin the text: v5 was still edited while it was being evaluated. Run
records therefore store, beside the version, a SHA-256 of each prompt as loaded here, which
matches the file in the Git history that produced the run. Files are read exactly as written,
including leading and trailing whitespace.
"""
import hashlib
from pathlib import Path

from config import settings

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
PROMPT_VERSION = settings.prompt_version

# The text of each prompt of the running version, as loaded at start-up (what the agents use).
_loaded: dict[str, str] = {}


def load_prompt(name: str, version: str = PROMPT_VERSION) -> str:
    path = PROMPTS_DIR / version / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt '{name}' not found for version '{version}' ({path}).")
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    if version == PROMPT_VERSION:
        _loaded[name] = text
    return text


def prompt_hashes() -> dict[str, str]:
    """SHA-256 of each prompt of the running version, as loaded, for the run record."""
    return {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in sorted(_loaded.items())}


ANALYSIS_AGENT_SYSTEM = load_prompt("analysis")
# Sent back with the failures when the analysis fails a blocking check ({failures}).
ANALYSIS_REVISION_REQUEST = load_prompt("analysis_revision_request")
# From v5 the reports are assembled in code; one agent writes the purchase request's narrative.
APPROVAL_AGENT_SYSTEM = load_prompt("approval_request")
EMAIL_DRAFT_AGENT_SYSTEM = load_prompt("email_draft")
# Email check and rewrite: {email} and {feedback} are filled in by services/agents.py.
EMAIL_VALIDATOR_SYSTEM = load_prompt("email_validator_system")
EMAIL_VALIDATOR_TEMPLATE = load_prompt("email_validator")
EMAIL_REVISION_SYSTEM_SUFFIX = load_prompt("email_revision_system_suffix")
EMAIL_REVISION_REQUEST_TEMPLATE = load_prompt("email_revision_request")
