import argparse
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree

import numpy as np
from chromboundary.data import csv_rows
from chromboundary.pack import write_dataset
from sources import NS, arrays, fetch_file

INDEX = Path(__file__).resolve().parent / "eoc"
DATASET = "MSV000084048"
TOLERANCE = 0.055


def run_file(row, cache):
    descriptor = quote(f"f.{DATASET}/{row['path']}", safe="/.")
    path = cache / DATASET / Path(row["path"]).name
    for host in ("proteomics2.ucsd.edu", "massive.ucsd.edu"):
        url = f"https://{host}/ProteoSAFe/DownloadResultFile?forceDownload=true&file={descriptor}"
        try:
            return fetch_file(url, path, row["sha256"])
        except (OSError, ValueError):
            if host == "massive.ucsd.edu":
                raise


def read_run(path):
    masses, curves = [], []
    for _, node in ElementTree.iterparse(path, events=("end",)):
        if node.tag != NS + "chromatogram":
            continue
        windows = [
            node.find(f"{NS}{kind}/{NS}isolationWindow/{NS}cvParam[@accession='MS:1000827']")
            for kind in ("precursor", "product")
        ]
        if all(v is not None for v in windows):
            masses.append([float(v.get("value")) for v in windows])
            curves.append(arrays(node, time_dtype=None))
        node.clear()
    if not masses:
        raise ValueError(f"No SRM chromatograms in {path}")
    return np.asarray(masses), curves


def prepare(run, cache):
    assays = defaultdict(list)
    for row in csv_rows(INDEX / "assays.csv", ("sequence", "charge", "light_q1", "heavy_q1")):
        pairs = tuple([float(row[c + "_q1"]), float(row[c + "_q3"])] for c in ("light", "heavy"))
        assays[row["sequence"], row["charge"]].append(pairs)
    rows = list(csv_rows(INDEX / "runs.csv", ("run", "sample", "path", "sha256")))
    if run is not None:
        rows = [r for r in rows if r["run"] == run]
    if not rows:
        raise ValueError("Unknown EOC acquisition")
    labels = {
        r["group_id"]: r
        for r in csv_rows(INDEX / "references.csv.gz", ("group_id", "label", "start_s", "end_s"))
    }
    for row in rows:
        masses, curves = read_run(run_file(row, cache))
        for (sequence, charge), definitions in assays.items():
            gid = "|".join(("Ovarian_cancer_dataset2", row["sample"], sequence, charge))
            pairs, used = [], set()
            for pair in definitions:
                traces = []
                for target in pair:
                    index = np.flatnonzero(np.all(np.abs(masses - target) <= TOLERANCE, axis=1))
                    if len(index) != 1 or int(index[0]) in used:
                        raise ValueError(f"Ambiguous or missing SRM transition for {gid}")
                    used.add(int(index[0]))
                    traces.append(curves[index[0]])
                pairs.append(traces)
            ref = labels[gid]
            yield (gid, "test", "EOC", ref["label"], ref["start_s"], ref["end_s"]), pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    parser.add_argument("--cache", type=Path, default=Path("datasets/source"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_dataset(args.out, prepare(args.run, args.cache))


if __name__ == "__main__":
    main()
