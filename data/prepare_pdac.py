import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

import joblib
from chromboundary.data import atomic_path, csv_rows
from chromboundary.pack import write_dataset

SOURCE = "https://zenodo.org/api/records/7964403/files/deep-mrm.zip/content"
CHECKSUM = "80ec14c1139059372947b5ba09bd408fcddb2eaf73208f7955d158162a497c59"
INDEX = Path(__file__).resolve().parent / "pdac_groups.csv.gz"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unpack(archive, path):
    with zipfile.ZipFile(archive) as bundle:
        (member,) = [
            i
            for i in bundle.namelist()
            if i.endswith("/PDAC_SIT_xic.pkl") and not i.startswith("__MACOSX/")
        ]
        with atomic_path(path) as temp:
            with bundle.open(member) as source, temp.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            if digest(temp) != CHECKSUM:
                raise ValueError("PDAC signal archive checksum mismatch")


def signals_file(source, cache):
    if source and source.suffix.lower() != ".zip":
        if digest(source) != CHECKSUM:
            raise ValueError("PDAC signal archive checksum mismatch")
        return source
    path = cache / "PDAC_SIT_xic.pkl"
    if path.is_file() and digest(path) == CHECKSUM:
        return path
    if source is None:
        source = cache / "deep-mrm.zip"
        with atomic_path(source) as temp:
            with urlopen(SOURCE, timeout=120) as response, temp.open("wb") as dest:
                shutil.copyfileobj(response, dest, 1 << 20)
            unpack(temp, path)
    else:
        unpack(source, path)
    return path


def prepare(source, split):
    data = joblib.load(source)
    for row in csv_rows(
        INDEX, ("group_id", "patient_id", "replicate_id", "peptide_id", "label", "split")
    ):
        if split is not None and row["split"] != split:
            continue
        key = int(row["patient_id"]), int(row["replicate_id"]), row["peptide_id"]
        sample = data[key]
        pairs = list(zip(sample["light"], sample["heavy"], strict=True))
        reference = row["group_id"], row["split"], "PDAC", row["label"], row["start_s"], row["end_s"]
        yield reference, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("datasets/source"))
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = signals_file(args.source, args.cache)
    write_dataset(args.out, prepare(path, args.split))


if __name__ == "__main__":
    main()
