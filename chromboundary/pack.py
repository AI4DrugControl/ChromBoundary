import csv
from pathlib import Path

import numpy as np

from .data import atomic_path


def write_dataset(directory, groups):
    output = Path(directory)
    ids, references = [], []
    group_offsets = [0]
    arrays = {c + "_" + k: [] for c in ("light", "heavy") for k in ("time_s", "intensity")}
    offsets = {c: [0] for c in ("light", "heavy")}
    for ref, pairs in groups:
        ids.append(ref[0])
        references.append(ref)
        if not pairs:
            raise ValueError(f"Missing transition pairs in {ref[0]}")
        group_offsets.append(group_offsets[-1] + len(pairs))
        for pair in pairs:
            for channel, (t, v) in zip(("light", "heavy"), pair, strict=True):
                if len(t) != len(v) or len(t) < 2:
                    raise ValueError(f"Invalid native chromatogram in {ref[0]}")
                arrays[channel + "_time_s"].append(t)
                arrays[channel + "_intensity"].append(v)
                offsets[channel].append(offsets[channel][-1] + len(t))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("The assay index is empty or contains duplicate groups")
    values = {k: np.concatenate(v) for k, v in arrays.items()}
    values.update({c + "_offsets": np.asarray(v, np.int64) for c, v in offsets.items()})
    values.update(group_id=np.asarray(ids), group_offsets=np.asarray(group_offsets, np.int64))
    with (
        atomic_path(output / "signals.npz") as signals,
        atomic_path(output / "references.csv") as labels,
    ):
        with signals.open("wb") as f:
            np.savez_compressed(f, **values)
        with labels.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(("group_id", "split", "panel", "label", "start_s", "end_s"))
            writer.writerows(references)
    print(f"{len(ids)} groups in {output}")
