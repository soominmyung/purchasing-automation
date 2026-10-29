"""
Agent prompts, loaded from prompts/<version>/.

Each version folder holds the whole set the pipeline uses, so a run is described by one
version name (settings.prompt_version), which is what run records store. A version is frozen
once runs or evaluation results have been recorded with it; changes go into a new folder, so
every recorded result can be traced to the exact prompt text that produced it. Files are read
exactly as written, including leading and trailing whitespace.
"""
from pathlib import Path

from config import settings

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
PROMPT_VERSION = settings.prompt_version


def load_prompt(name: str, version: str = PROMPT_VERSION) -> str:
    path = PROMPTS_DIR / version / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt '{name}' not found for version '{version}' ({path}).")
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


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
