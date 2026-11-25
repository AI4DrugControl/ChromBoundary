import argparse
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree

from chromboundary.data import csv_rows
from chromboundary.pack import write_dataset
from openpyxl import load_workbook
from sources import NS, arrays, fetch_file

SOURCE = "https://gitlab.gwdg.de/api/v4/projects/18737/repository/files/"
REVISION = "c57f03d248097a70a5795195c7380d256a920381"
INDEX = Path(__file__).resolve().parent / "hilic"


def fetch(relative, cache, digest):
    url = f"{SOURCE}{quote(relative, safe='')}/raw?ref={REVISION}"
    return fetch_file(url, cache / relative, digest)


def chromatograms(path, wanted):
    curves = {}
    for _, node in ElementTree.iterparse(path, events=("end",)):
        if node.tag != NS + "chromatogram":
            continue
        name = node.get("id")
        if name in wanted:
            if name in curves:
                raise ValueError(f"Ambiguous chromatogram {name}")
            curves[name] = arrays(node)
        node.clear()
    if curves.keys() != wanted:
        raise ValueError(f"Missing chromatograms in {path}")
    return curves


def reference_sheet(path, panel):
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book.active
        sheet.reset_dimensions()
        table = list(sheet.values)
    finally:
        book.close()
    markers, fields = table[:2]
    column = fields.index("Data File")
    runs = {Path(r[column]).stem: r for r in table[2:] if r[column]}
    columns = {}
    for i, name in enumerate(markers):
        if not isinstance(name, str) or not name.strip() or name.startswith("Qualifier"):
            continue
        if panel == "HILIC2" and fields[i] == "Area":
            area, start, end = [
                i + fields[i : i + 5].index(key) for key in ("Area", "Int. Start", "Int. End")
            ]
            columns[name.strip()] = (area, start, end)
        if panel == "HILIC1a" and fields[i] == "Int. Start":
            columns[name.strip()] = (i - 1, i, i + 1)
    return runs, columns


def reference(row, cols, panel):
    value, lo, hi = (float(row[i]) if row[i] is not None else None for i in cols)
    if panel == "HILIC2" and value == 0:
        return 0, "", ""
    if panel == "HILIC1a" and (value, lo, hi) == (0, 0, 0):
        return -1, "", ""
    if (
        (panel == "HILIC2" and (value is None or value <= 0))
        or lo is None
        or hi is None
        or not 0 <= lo < hi
    ):
        raise ValueError("Invalid published reference interval")
    return 1, 60 * lo, 60 * hi


def prepare(panels, split, cache):
    hashes = {r["path"]: r["sha256"] for r in csv_rows(INDEX / "checksums.csv", ("path", "sha256"))}
    by_run = defaultdict(list)
    for row in csv_rows(INDEX / "groups.csv.gz", ("panel", "run_id", "split", "light_id", "heavy_id")):
        if row["panel"] in panels and (split is None or row["split"] == split):
            by_run[row["panel"], row["run_id"]].append(row)
    for panel in panels:
        workbook = "trainingSolution.xlsx" if panel == "HILIC2" else "training_solution.xlsx"
        key = f"{panel}/{workbook}"
        runs, cols = reference_sheet(fetch(key, cache, hashes[key]), panel)
        for (p, run), items in by_run.items():
            if p != panel:
                continue
            key = f"{panel}/{run}.mzML"
            ids = {r[c] for r in items for c in ("light_id", "heavy_id")}
            curves = chromatograms(fetch(key, cache, hashes[key]), ids)
            for row in items:
                label, lo, hi = reference(runs[run], cols[row["analyte"]], panel)
                ref = row["group_id"], row["split"], panel, label, lo, hi
                yield ref, [(curves[row["light_id"]], curves[row["heavy_id"]])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("HILIC2", "HILIC1a", "both"), default="HILIC2")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--cache", type=Path, default=Path("datasets/source"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    panels = ("HILIC2", "HILIC1a") if args.panel == "both" else (args.panel,)
    write_dataset(args.out, prepare(panels, args.split, args.cache))


if __name__ == "__main__":
    main()
