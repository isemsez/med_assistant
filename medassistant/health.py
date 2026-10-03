"""Health checks: the three LLM endpoints, index freshness, PDF corpus.

Designed for /api/health and the canary CLI. Endpoint probes use plain httpx
with a short timeout and NO retries — a health check must fail fast. Results
are cached for HEALTH_CACHE_S seconds so the admin page can poll without
hammering the models (a cache_ttl of 0 disables caching).
"""

import datetime
import json
import os
import time
from pathlib import Path

import httpx

from . import config, gindex

CACHE_TTL = float(os.environ.get("HEALTH_CACHE_S", "30"))
_cache = {}  # key -> (monotonic ts, value)


def _ts():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _cached(key, fn, ttl=None):
    ttl = CACHE_TTL if ttl is None else ttl
    if ttl <= 0:
        return fn()
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def _headers(endpoint):
    h = {"Content-Type": "application/json"}
    if endpoint.get("api_key"):
        h["Authorization"] = f"Bearer {endpoint['api_key']}"
    return h


def check_model(endpoint, timeout=5.0):
    """GET {base}/models must list the configured model."""
    if not endpoint.get("base_url") or not endpoint.get("model"):
        return {"ok": False, "error": "endpoint not configured"}
    url = f"{endpoint['base_url']}/models"
    try:
        r = httpx.get(url, headers=_headers(endpoint), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        entries = data.get("data") or data.get("models") or []
        ids = {e.get("id") or e.get("name") for e in entries}
        # config names are often aliases without the tag (medgemma-cpu ==
        # medgemma-cpu:latest for Ollama); match those too
        if not (endpoint["model"] in ids
                or any(i and i.startswith(endpoint["model"] + ":") for i in ids)):
            return {"ok": False, "error": f"model {endpoint['model']!r} not in "
                                          f"{sorted(i for i in ids if i)[:10]}"}
        return {"ok": True}
    except (httpx.HTTPError, OSError, ValueError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def check_embed(endpoint, timeout=20.0):
    """A one-token embed ping must return a vector."""
    if not endpoint.get("base_url") or not endpoint.get("model"):
        return {"ok": False, "error": "endpoint not configured"}
    url = f"{endpoint['base_url']}/embeddings"
    try:
        r = httpx.post(url, headers=_headers(endpoint), timeout=timeout,
                       content=json.dumps({"model": endpoint["model"],
                                           "input": ["пинг"]}))
        r.raise_for_status()
        data = r.json()
        vecs = data.get("data") or []
        if not vecs or not (vecs[0].get("embedding") or []):
            return {"ok": False, "error": "empty embedding in response"}
        return {"ok": True, "dim": len(vecs[0]["embedding"])}
    except (httpx.HTTPError, OSError, ValueError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def check_index(index_dir=None, docs_meta_path=None):
    """Index must exist and its build-time docs_meta hash must match current.

    fresh=None means the index predates freshness stamps (legacy build) —
    reported as ok with a hint to rebuild.
    """
    index_dir = Path(index_dir or config.INDEX_DIR)
    docs_meta_path = Path(docs_meta_path or config.DOCS_META_PATH)
    meta_path = index_dir / "meta.json"
    if not meta_path.is_file():
        return {"ok": False, "fresh": None, "error": "index not built "
                f"({meta_path} missing) — run: python -m medassistant build-index"}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        docs = config.load_docs_meta(docs_meta_path)
    except (OSError, ValueError) as e:
        return {"ok": False, "fresh": None, "error": f"{type(e).__name__}: {e}"}
    out = {"docs_meta": len(docs), "vectors": len(meta.get("rows") or [])}
    stamped = meta.get("docs_meta_sha1")
    if not stamped:
        return {"ok": True, "fresh": None, **out,
                "note": "legacy index without freshness stamp — rebuild advised"}
    out["fresh"] = stamped == gindex.docs_hash(docs)
    out["ok"] = out["fresh"]
    if not out["fresh"]:
        out["error"] = ("docs_meta.json changed after build-index "
                        f"({meta.get('docs_meta_count')} -> {len(docs)} guidelines)"
                        " — rebuild: python -m medassistant build-index")
    return out


def check_corpus(clinrec_dir=None):
    """The local guideline PDF folder must exist and be non-empty."""
    d = Path(clinrec_dir or config.CLINREC_DIR)
    if not d.is_dir():
        return {"ok": False, "pdfs": 0, "error": f"{d} missing"}
    n = sum(1 for _ in d.glob("*.pdf"))
    return {"ok": n > 0, "pdfs": n, **({} if n else
                                        {"error": "no PDFs in corpus dir"})}


def run_health(force=False, cache_ttl=None):
    """All checks -> {"ok": bool, "ts", "checks": {name: check}}."""
    checks = {
        "answerer": lambda: _cached("answerer", lambda: check_model(config.CANDIDATE),
                                    ttl=cache_ttl),
        "labeler": lambda: _cached("labeler", lambda: check_model(config.LABELER),
                                   ttl=cache_ttl),
        "embed": lambda: _cached("embed", lambda: check_embed(config.EMBED),
                                 ttl=cache_ttl),
        "index": lambda: check_index(),
        "corpus": lambda: _cached("corpus", check_corpus, ttl=cache_ttl),
    }
    if force:
        _cache.clear()
    out = {name: fn() for name, fn in checks.items()}
    return {"ok": all(c.get("ok") for c in out.values()), "ts": _ts(), "checks": out}
