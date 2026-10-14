"""Patient cards (амбулаторные карты) — example-only per-question prompt
context (architecture.md §9, owner decision 2026-10-06; the full chart as
the attachable card is the owner decision 2026-10-14).

A card is a hand-made ``*.амбулаторная_карта.md`` file in CARD_DIR. The
physician picks one in the UI; its text is appended to the answerer's user
message for that question only. Nothing is stored — the stored user message
stays the plain question, and the card text appears only in the LLM call.
Shared by the server and the CLI so both validate identically.
"""

from pathlib import Path

from . import config

SUFFIX = ".амбулаторная_карта.md"


class CardError(ValueError):
    """Validation failure with a physician-facing Russian message."""


def display_name(filename):
    """'Иванов_ИИ.амбулаторная_карта.md' -> 'Иванов ИИ' (suffix off, underscores → spaces)."""
    return filename[: -len(SUFFIX)].replace("_", " ")


def list_cards(card_dir=None):
    """Cards in the directory, sorted by display name; missing dir -> []."""
    d = Path(card_dir) if card_dir is not None else config.CARD_DIR
    if not d.is_dir():
        return []
    out = []
    for p in d.glob(f"*{SUFFIX}"):
        if p.is_file():
            st = p.stat()
            out.append({"file": p.name, "name": display_name(p.name),
                        "size": st.st_size, "mtime": int(st.st_mtime)})
    out.sort(key=lambda c: (c["name"], c["file"]))
    return out


def _read_checked(path, label, max_chars):
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CardError(f"Карта пациента «{label}» не найдена.")
    except UnicodeError:
        raise CardError(f"Карта пациента «{label}» не является текстовым "
                        "файлом в кодировке UTF-8.")
    except OSError as e:
        raise CardError(f"Не удалось прочитать карту пациента «{label}»: {e}")
    limit = config.CARD_MAX_CHARS if max_chars is None else max_chars
    if len(text) > limit:
        raise CardError(
            f"Карта пациента «{label}» слишком велика: {len(text)} символов "
            f"(максимум {limit}). Сократите карту.")
    return text


def load_card(filename, card_dir=None, max_chars=None):
    """A plain filename inside CARD_DIR -> (display_name, text).

    The file is re-read fresh on every call — the doctor may edit it
    mid-session. Raises CardError on any violation: traversal-safe name only
    (`Path(...).name` plus a parent check — a suffix check alone is not
    enough), mandatory card suffix, existence, size within the limit.
    """
    d = (Path(card_dir) if card_dir is not None else config.CARD_DIR).resolve()
    name = Path(filename).name
    if (not name or name in {".", ".."} or Path(name).name != name
            or not name.endswith(SUFFIX)):
        raise CardError(f"Укажите файл карты пациента («*{SUFFIX}») "
                        "из папки карт.")
    p = d / name
    if p.parent != d or p.resolve().parent != d:
        raise CardError("Недопустимое имя файла карты пациента.")
    return display_name(name), _read_checked(p, name, max_chars)


def load_card_path(path, max_chars=None):
    """Explicit-path variant for the CLI: same suffix and size rules."""
    p = Path(path)
    if not p.name.endswith(SUFFIX):
        raise CardError(f"Карта пациента должна быть файлом «*{SUFFIX}».")
    return display_name(p.name), _read_checked(p, str(p), max_chars)
