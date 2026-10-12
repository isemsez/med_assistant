"""Import the GRLS register export (XLSX) into data/drugs.sqlite (setup tool).

Reads the official ГРЛС daily export (or a user-trimmed copy of it) — an
XLSX whose header row carries «Торговое наименование», «Международное
непатентованное или химическое наименование», «Фармако-терапевтическая
группа» — and writes a faithful row-for-row copy into a small SQLite DB
used as the offline Russian drugs reference. No dedup, no normalization:
the import keeps every register row and only the three source columns.

Stdlib only (zipfile + ElementTree + sqlite3); openpyxl is deliberately
not required. One-time setup utility, not part of the runtime package.

Usage:
    python scripts/grls_to_sqlite.py <export.xlsx> [--db data/drugs.sqlite]
"""

import argparse
import re
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
TRADE_HEADER_PREFIX = "Торговое наименование"

COLUMNS = [
    ("A", "trade_name", "Торговое наименование лекарственного препарата"),
    ("B", "inn", "Международное непатентованное или химическое наименование"),
    ("C", "pharm_group", "Фармако-терапевтическая группа"),
]


def _load_shared_strings(zf):
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [
        "".join(t.text or "" for t in si.iter(M + "t"))
        for si in root.findall(M + "si")
    ]


def _first_worksheet_path(zf):
    names = zf.namelist()
    for name in sorted(names):
        if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name):
            return name
    raise ValueError("no xl/worksheets/sheet*.xml found in the XLSX")


def _cell_value(cell, shared):
    t = cell.get("t")
    v = cell.find(M + "v")
    if t == "inlineStr":
        is_el = cell.find(M + "is")
        if is_el is None:
            return ""
        return "".join(x.text or "" for x in is_el.iter(M + "t"))
    if v is None or v.text is None:
        return ""
    if t == "s":
        return shared[int(v.text)]
    return v.text


def _rows(path):
    """Yield {column_letter: text} dicts for every row of the first sheet."""
    with zipfile.ZipFile(path) as zf:
        shared = _load_shared_strings(zf)
        sheet = ET.fromstring(zf.read(_first_worksheet_path(zf)))
        for row in sheet.iter(M + "row"):
            cells = {}
            for cell in row.findall(M + "c"):
                ref = cell.get("r") or ""
                m = re.match(r"([A-Z]+)", ref)
                if not m:
                    continue
                cells[m.group(1)] = _cell_value(cell, shared)
            yield cells


def _clean(value):
    if value is None:
        return None
    value = value.strip()
    return value or None


def import_xlsx(xlsx_path, db_path):
    data_rows = []
    started = False
    saw_header = False
    for cells in _rows(xlsx_path):
        if not started:
            if (cells.get("A") or "").startswith(TRADE_HEADER_PREFIX):
                started, saw_header = True, True
            continue
        values = [_clean(cells.get(letter)) for letter, _, _ in COLUMNS]
        if any(v is not None for v in values):
            data_rows.append(values)
    if not saw_header:
        raise ValueError(f"header row starting with «{TRADE_HEADER_PREFIX}…» not found")

    export_date_m = re.search(r"(\d{4}-\d{2}-\d{2})", Path(xlsx_path).name)
    export_date = export_date_m.group(1) if export_date_m else None

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute("DROP TABLE IF EXISTS drugs")
            con.execute(
                """
                CREATE TABLE drugs (
                    trade_name  TEXT,
                    inn         TEXT,
                    pharm_group TEXT
                )
                """
            )
            con.executemany(
                "INSERT INTO drugs (trade_name, inn, pharm_group) VALUES (?, ?, ?)",
                data_rows,
            )
            con.execute("CREATE INDEX idx_drugs_trade_name ON drugs(trade_name)")
            con.execute("CREATE INDEX idx_drugs_inn ON drugs(inn)")
            con.execute("DROP TABLE IF EXISTS meta")
            con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            con.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [
                    ("source_file", Path(xlsx_path).name),
                    ("source_export_date", export_date),
                    ("imported_rows", str(len(data_rows))),
                    ("columns", ", ".join(name for _, name, _ in COLUMNS)),
                    ("imported_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
                ],
            )
    finally:
        con.close()

    print(f"imported {len(data_rows)} rows -> {db_path}")
    for i, (_, name, title) in enumerate(COLUMNS):
        filled = sum(1 for r in data_rows if r[i] is not None)
        print(f"  {name:<11} non-null: {filled:>6}  ({title})")
    return len(data_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xlsx", help="GRLS export XLSX (header row «Торговое наименование…»)")
    parser.add_argument("--db", default="data/drugs.sqlite", help="output SQLite path")
    args = parser.parse_args(argv)

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.is_file():
        parser.error(f"xlsx not found: {xlsx_path}")
    import_xlsx(xlsx_path, Path(args.db))


if __name__ == "__main__":
    sys.exit(main())
