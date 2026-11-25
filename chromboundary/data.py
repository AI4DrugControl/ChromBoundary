import csv
import gzip
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Record:
    group_id: str
    grid: np.ndarray
    signal: np.ndarray
    profile: str


@dataclass(frozen=True)
class Reference:
    label: int
    bounds: tuple | None
    split: str
    panel: str


@contextmanager
def atomic_path(destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_rows(path, required):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        yield from reader


def align(group_id, pairs, profile):
    if not pairs:
        raise ValueError(f"No transition pairs in {group_id}")
    traces = [pair[channel] for channel in (0, 1) for pair in pairs]
    for clock, intensity in traces:
        if (
            clock.ndim != 1
            or len(clock) < 2
            or clock.shape != intensity.shape
            or not np.isfinite(clock).all()
            or not np.isfinite(intensity).all()
            or np.any(np.diff(clock) <= 0)
            or np.any(intensity < 0)
        ):
            raise ValueError(f"Invalid retention times or intensities in {group_id}")
    first = max(float(t[0]) for t, _ in traces)
    last = min(float(t[-1]) for t, _ in traces)
    if last < first and profile == "peptide":
        first = min(float(t[0]) for t, _ in traces)
        last = max(float(t[-1]) for t, _ in traces)
    count = int((last - first) / 0.5)
    if count < 4:
        raise ValueError(f"Insufficient acquisition range in {group_id}")
    grid = np.linspace(first, last, count, dtype=np.float32)
    values = [
        np.interp(grid, t, y, left=0, right=0) if profile == "peptide" else np.interp(grid, t, y)
        for t, y in traces
    ]
    signal = np.asarray(values, dtype=np.float32).reshape(2, len(pairs), count)
    if profile == "metabolite":
        signal /= max(float(signal.max()), 1e-12)
    return Record(group_id, grid, signal, profile)


def tabular_records(path, profile, wanted):
    groups = {}
    for row in csv_rows(path, ("group_id", "transition", "channel", "time_s", "intensity")):
        gid = row["group_id"]
        if wanted is not None and gid not in wanted:
            continue
        transition, channel = row["transition"], row["channel"]
        if not gid or not transition or channel not in ("light", "heavy"):
            raise ValueError(
                "Each observation needs a group, a transition and a light or heavy channel"
            )
        pair = groups.setdefault(gid, {}).setdefault(transition, {})
        pair.setdefault(channel, []).append((float(row["time_s"]), float(row["intensity"])))
    for gid, transitions in groups.items():
        pairs = []
        for transition, channels in transitions.items():
            if set(channels) != {"light", "heavy"}:
                raise ValueError(f"Incomplete transition {transition} in {gid}")
            pairs.append(tuple(np.asarray(sorted(channels[c])).T for c in ("light", "heavy")))
        yield align(gid, pairs, profile)


def packed_records(path, profile, wanted):
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["group_id"].tolist()
        offsets = archive["group_offsets"]
        arrays = {k: archive[k] for k in archive.files if k.startswith(("light_", "heavy_"))}
    if (
        len(set(ids)) != len(ids)
        or len(offsets) != len(ids) + 1
        or offsets[0] != 0
        or np.any(np.diff(offsets) < 1)
    ):
        raise ValueError("Invalid group index in signal archive")
    for channel in ("light", "heavy"):
        edges = arrays[channel + "_offsets"]
        t, y = arrays[channel + "_time_s"], arrays[channel + "_intensity"]
        if (
            len(edges) != offsets[-1] + 1
            or edges[0] != 0
            or edges[-1] != len(t)
            or len(t) != len(y)
            or np.any(np.diff(edges) < 2)
        ):
            raise ValueError(f"Invalid {channel} index in signal archive")
    for i, gid in enumerate(ids):
        if wanted is not None and gid not in wanted:
            continue
        pairs = []
        for j in range(offsets[i], offsets[i + 1]):
            pair = []
            for channel in ("light", "heavy"):
                lo, hi = arrays[channel + "_offsets"][j : j + 2]
                pair.append((arrays[channel + "_time_s"][lo:hi], arrays[channel + "_intensity"][lo:hi]))
            pairs.append(pair)
        yield align(gid, pairs, profile)


def records(path, profile, wanted=None):
    if profile not in ("peptide", "metabolite"):
        raise ValueError("Unknown acquisition profile")
    path = Path(path)
    if path.is_dir():
        parts = sorted(path.glob("signals-*.npz"))
        if not parts and (path / "signals.npz").is_file():
            parts = [path / "signals.npz"]
    else:
        parts = [path]
    seen = set()
    for part in parts:
        reader = packed_records if part.suffix == ".npz" else tabular_records
        for record in reader(part, profile, wanted):
            if record.group_id in seen:
                raise ValueError(f"Repeated signal group {record.group_id}")
            seen.add(record.group_id)
            yield record
    if not seen:
        raise ValueError("No chromatogram groups selected")
    if wanted is not None and seen != wanted:
        raise ValueError(f"Missing signals for {len(wanted - seen)} requested groups")


def read_references(path):
    references = {}
    for row in csv_rows(path, ("group_id", "label", "start_s", "end_s")):
        gid, label = row["group_id"], int(row["label"])
        if not gid or gid in references or label not in (-1, 0, 1):
            raise ValueError(f"Invalid or duplicate reference {gid}")
        bounds = None
        if label == 1:
            bounds = float(row["start_s"]), float(row["end_s"])
            if not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
                raise ValueError(f"Invalid reference boundaries in {gid}")
        references[gid] = Reference(label, bounds, row.get("split", ""), row.get("panel", ""))
    if not references:
        raise ValueError("Reference file has no groups")
    return references


def collate(records, references=None, device="cpu"):
    batch = len(records)
    channels = max(item.signal.shape[1] for item in records)
    length = max(len(item.grid) for item in records)
    x = np.zeros((batch, 2, channels, length), np.float32)
    cm = np.zeros((batch, channels), bool)
    tm = np.zeros((batch, length), bool)
    targets = {"time_mask": tm, "channel_mask": cm}
    if references is not None:
        targets.update(
            mask=np.zeros((batch, length), np.float32),
            has_peak=np.zeros(batch, np.float32),
            starts=np.zeros(batch, np.float32),
            ends=np.zeros(batch, np.float32),
        )
    for i, record in enumerate(records):
        _, c, n = record.signal.shape
        x[i, :, :c, :n] = (
            record.signal
            if record.profile == "metabolite"
            else record.signal / max(float(record.signal.max()), 1e-12)
        )
        cm[i, :c], tm[i, :n] = True, True
        if references is None:
            continue
        ref = references[record.group_id]
        if ref.label == -1:
            raise ValueError(f"Unknown training label for {record.group_id}")
        if ref.bounds is not None:
            lo, hi = ref.bounds
            grid = record.grid
            if record.profile == "metabolite" and (lo < grid[0] - 1e-4 or hi > grid[-1] + 1e-4):
                raise ValueError(f"Reference outside the acquisition grid in {record.group_id}")
            start, end = np.interp([lo, hi], grid, np.arange(n)).astype(np.float32)
            if int(start) >= int(end):
                raise ValueError(f"Reference shorter than a scan interval in {record.group_id}")
            targets["starts"][i], targets["ends"][i] = start, end
            targets["has_peak"][i] = 1
            targets["mask"][i, int(start) : int(end) + 1] = 1
    return torch.from_numpy(x).to(device), {
        k: torch.from_numpy(v).to(device) for k, v in targets.items()
    }


def overlap(first, second):
    a, b = first[:2]
    c, d = second[:2]
    intersection = max(0.0, min(b, d) - max(a, c))
    return intersection / (b - a + d - c - intersection)
