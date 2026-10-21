"""Renderer: picked guideline ids -> appendix block; failure notes (architecture.md §3[5], §5).

Pure Python. ids -> docs_meta -> local guideline PDFs (CLINREC_DIR, served by
the chat server at /data/pdf). The appendix is appended under the untouched
raw answer as a verification block for the physician.
"""

from pathlib import Path
from urllib.parse import quote

PDF_URL_PREFIX = "/data/pdf/"

NOTE_EXTRACT_ERROR = "Сбой распознавания заболевания — клинические рекомендации не подобраны."
NOTE_NO_DISEASE = "Заболеваний в ответе не найдено."
NOTE_NO_DISEASE_MANUAL = "Заболеваний в запросе не найдено."
NOTE_SEARCH_ERROR = "Сбой подбора рекомендаций."
NOTE_SELECT_ERROR = "Сбой подбора рекомендаций."

APPENDIX_HEADER = "Клинические рекомендации Минздрава России (для проверки):"


def filtered_out(by_disease, cand_by_disease):
    """Flat list of candidates not verified for any disease, deduped by id.

    by_disease: [{disease, picks}] — the verified picks; cand_by_disease:
    [(disease, [{code_version, name, url[, path]}])] — per-disease resolved
    candidates. A candidate picked for ANY disease counts as verified. Each
    filtered entry keeps the disease whose search produced it.
    """
    picked = {p["code_version"] for g in by_disease for p in g["picks"]}
    seen, out = set(), []
    for disease, cands in cand_by_disease:
        for c in cands:
            cv = c["code_version"]
            if cv in picked or cv in seen:
                continue
            seen.add(cv)
            out.append({**c, "disease": disease})
    return out


def file_title(file, code_version):
    """'115_3_Сахарный диабет.pdf' + '115_3' -> 'Сахарный диабет'."""
    stem = Path(file).stem
    prefix = f"{code_version}_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def resolve_picks(picks, docs_meta, clinrec_dir=None):
    """[{code_version}] -> [{code_version, name, url[, path]}] in pick order.

    Name and link come from the local PDF file name (docs_meta `file`): name is
    the textual part after the `{code_version}_` prefix, url is a /data/pdf/<file>
    link for the chat UI, path is the absolute file path for the CLI appendix
    text. Rows without `file` fall back to meta name/url. With clinrec_dir given,
    picks whose PDF is missing on disk are dropped (never a dead link); no disk
    access happens without it. Unknown ids are dropped (never rendered from
    model data alone).
    """
    root = Path(clinrec_dir) if clinrec_dir else None
    out = []
    for p in picks:
        cv = p.get("code_version") if isinstance(p, dict) else p
        meta = docs_meta.get(cv)
        if not meta:
            continue
        file = (meta.get("file") or "").strip()
        if file:
            if root is not None:
                path = root / file
                if not path.is_file():
                    continue
                pick = {"code_version": cv,
                        "name": file_title(file, cv) or (meta.get("name") or "").strip(),
                        "url": PDF_URL_PREFIX + quote(file),
                        "path": str(path)}
            else:
                pick = {"code_version": cv,
                        "name": file_title(file, cv) or (meta.get("name") or "").strip(),
                        "url": PDF_URL_PREFIX + quote(file)}
        else:
            pick = {"code_version": cv,
                    "name": (meta.get("name") or "").strip(),
                    "url": meta.get("url") or ""}
        # the register audience rides on the pick (owner catch
        # 2026-10-21): same-NAME adult/pediatric pairs exist and the
        # title alone never says which — the renderer marks pediatric
        # picks so the doctor can see it
        age = (meta.get("age") or "").strip()
        if age:
            pick["age"] = age
        out.append(pick)
    return out


def render_appendix(by_disease):
    """by_disease: [{disease, picks: [{code_version, name, url[, path]}]}] -> str | None.

    One block: header, then per disease its guideline links. The absolute local
    path is preferred over the /data/pdf URL (CLI/audit text); old payloads
    without path fall back to url. Returns None when no disease has picks
    (caller shows per-disease notes instead).
    """
    groups = [g for g in by_disease if g.get("picks")]
    if not groups:
        return None
    lines = [APPENDIX_HEADER]
    multi = len(groups) > 1
    for g in groups:
        if multi:
            lines.append(f"«{g['disease']}»:")
        for p in g["picks"]:
            # «, дети» only for pediatric-only docs («Дети» audience):
            # attaching one to an adult question is the silent-wrong-doc
            # hazard; adult and mixed docs go unmarked to keep lines short
            kids = ", дети" if p.get("age") == "Дети" else ""
            lines.append(f"— «{p['name']}» ({p['code_version']}{kids}): "
                         f"{p.get('path') or p['url']}")
    return "\n".join(lines)
