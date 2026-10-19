import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "DOCS_META": "data/docs_meta.json",
    "ABBR_PATH": "data/abbr2term.yaml",
    "INDEX_DIR": "data/index",
    "CLINREC_DIR": "data/clinrec_minzdrav",
    "SESSIONS_DB": "data/sessions.sqlite",
    "AUDIT_LOG": "logs/audit.jsonl",
    "CARD_DIR": "data/patient_cards",
    "DRUGS_DB": "data/drugs.sqlite",
    "DRUG_ALIASES": "data/drug_aliases.yaml",
    "INN_GLOSS": "data/inn_gloss.yaml",
}


def load_env(path=None):
    p = Path(path or PROJECT_ROOT / ".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()


def _path(env_key):
    v = os.environ.get(env_key, DEFAULTS[env_key])
    p = Path(v)
    return p if p.is_absolute() else PROJECT_ROOT / p


DOCS_META_PATH = _path("DOCS_META")
ABBR_PATH = _path("ABBR_PATH")
INDEX_DIR = _path("INDEX_DIR")
CLINREC_DIR = _path("CLINREC_DIR")
SESSIONS_DB = _path("SESSIONS_DB")
AUDIT_LOG_PATH = _path("AUDIT_LOG")
CARD_DIR = _path("CARD_DIR")
DRUGS_DB = _path("DRUGS_DB")
DRUG_ALIASES_PATH = _path("DRUG_ALIASES")
INN_GLOSS_PATH = _path("INN_GLOSS")

HISTORY_WINDOW = int(os.environ.get("HISTORY_WINDOW", "6"))
DISEASE_CAP = int(os.environ.get("DISEASE_CAP", "3"))
TOP_K = int(os.environ.get("TOP_K", "5"))

# Housekeeping (owner decision 2026-10-02: laptop-only, built-in): rotate the
# JSONL audit log by size; timestamped backups of the sessions DB at server
# startup (physician questions are sensitive — backups stay on this machine).
AUDIT_ROTATE_MB = float(os.environ.get("AUDIT_ROTATE_MB", "10"))
AUDIT_KEEP = int(os.environ.get("AUDIT_KEEP", "3"))
SESSIONS_BACKUP_KEEP = int(os.environ.get("SESSIONS_BACKUP_KEEP", "5"))

# Auto-pick (owner decision 2026-09-29): when one search candidate clearly
# dominates, skip labeler call 2 and take it. AUTO_PICK=0 disables.
AUTO_PICK = os.environ.get("AUTO_PICK", "1") not in {"0", "false", "no"}
AUTO_PICK_MIN = float(os.environ.get("AUTO_PICK_MIN", "0.80"))
AUTO_PICK_GAP = float(os.environ.get("AUTO_PICK_GAP", "0.05"))

# Patient cards (owner decision 2026-10-06, architecture.md §9): hand-made
# *.амбулаторная_карта.md files, per-question prompt context, nothing stored.
CARD_MAX_CHARS = int(os.environ.get("CARD_MAX_CHARS", "8000"))

# Register-grounded МНН lookup (owner decision 2026-10-12, architecture.md
# §3[1]): trade names from drugs.sqlite get a «справочно» P.S. + a persisted
# `.sources.drug` block under the question bubble. Matching v2 (owner
# decision 2026-10-13): first-word dictionary keys, adjacent-bigram
# restriction, modal МНН, then curated aliases (data/drug_aliases.yaml —
# owner-curated truth, value = register trade name or a direct МНН for
# register holes), only when nothing matched. A
# missing DB silently disables the feature; DRUG_PS_MAX caps matches.
DRUG_PS_MAX = int(os.environ.get("DRUG_PS_MAX", "5"))

ANSWERER_MAX_TOKENS = int(os.environ.get("ANSWERER_MAX_TOKENS", "3072"))
LABELER_EXTRACT_MAX_TOKENS = int(os.environ.get("LABELER_EXTRACT_MAX_TOKENS", "200"))
LABELER_SELECT_MAX_TOKENS = int(os.environ.get("LABELER_SELECT_MAX_TOKENS", "128"))

# Structured output transport (owner decision 2026-10-18, llama.cpp backend):
# llama-server's response_format json_schema/json_object crashes on
# Gemma-style vocabularies (autoparser grammar cannot accept the " model"
# leading-space piece during prompt prefill). Two working transports:
#   none            - no response_format: the labeler thinks (reasoning is
#                     split server-side), parse_json_object tolerates fences
#                     and prose — production parity
#                     (the shipped default)
#   grammar         - GBNF via schema_to_grammar in the request-level
#                     "grammar" field: guaranteed-valid JSON but no thinking
#                     (the grammar forces the first token), which measurably
#                     weakens candidate selection (canary 2026-10-18)
#   response_format - OpenAI json_schema (llama.cpp and other servers)
STRUCTURED_OUTPUT = os.environ.get("STRUCTURED_OUTPUT", "response_format")
if STRUCTURED_OUTPUT not in {"none", "grammar", "response_format"}:
    raise SystemExit(f"STRUCTURED_OUTPUT must be none|grammar|response_format, "
                     f"got {STRUCTURED_OUTPUT!r}")

# Answerer thinking-loop watchdog (owner decision 2026-10-06): abort a
# degenerate pre-content reasoning loop, retry once with an
# anti-repetition nudge. LOOP_GUARD=0 disables the watchdog.
LOOP_GUARD = os.environ.get("LOOP_GUARD", "1") not in {"0", "false", "no"}
LOOP_MIN_UNIT = int(os.environ.get("LOOP_MIN_UNIT", "8"))
LOOP_REPEATS = int(os.environ.get("LOOP_REPEATS", "3"))
LOOP_WINDOW_CHARS = int(os.environ.get("LOOP_WINDOW_CHARS", "4096"))
LOOP_RETRIES = int(os.environ.get("LOOP_RETRIES", "1"))
LOOP_RETRY_FREQ_PENALTY = float(os.environ.get("LOOP_RETRY_FREQ_PENALTY", "0.5"))


def _endpoint(prefix):
    extra = os.environ.get(f"{prefix}_EXTRA_BODY", "").strip()
    return {
        "base_url": os.environ.get(f"{prefix}_BASE_URL", "").rstrip("/"),
        "api_key": os.environ.get(f"{prefix}_API_KEY", ""),
        "model": os.environ.get(f"{prefix}_MODEL", ""),
        "extra_body": json.loads(extra) if extra else {},
    }


CANDIDATE = _endpoint("CANDIDATE")
LABELER = _endpoint("LABELER")
EMBED = _endpoint("EMBED")


def require(endpoint, name):
    if not endpoint["base_url"] or not endpoint["model"]:
        raise SystemExit(
            f"{name} endpoint not configured. Set {name}_BASE_URL and {name}_MODEL"
            f"{' and ' + name + '_API_KEY' if not endpoint['api_key'] else ''} in .env"
        )
    return endpoint


def load_docs_meta(path=None):
    """docs_meta.json -> {code_version: meta} for all guidelines (no filters)."""
    p = Path(path or DOCS_META_PATH)
    rows = json.loads(p.read_text(encoding="utf-8"))
    return {row["code_version"]: row for row in rows}
