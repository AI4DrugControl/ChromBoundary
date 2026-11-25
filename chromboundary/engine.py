import csv
import gzip
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from .data import atomic_path, collate, csv_rows
from .decode import decode_boundaries
from .model import ChromBoundaryNet
from .refine import feature_pair, probability

PRESETS = ("pdac", "hilic2", "hilic_joint")


def load_checkpoint(name, device):
    path = (
        Path(__file__).resolve().parent / "weights" / (name + ".pt") if name in PRESETS else Path(name)
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1 or state.get("profile") not in ("peptide", "metabolite"):
        raise ValueError(f"Invalid ChromBoundary checkpoint {path}")
    model = ChromBoundaryNet()
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state["profile"], state["refiner"]


def batches(records, size):
    if size < 1:
        raise ValueError("Batch size must be positive")
    items = iter(records)
    while batch := list(islice(items, size)):
        yield batch


def predict_raw(model, records, batch_size, device):
    model.eval()
    for batch in batches(records, batch_size):
        with torch.inference_mode():
            x, target = collate(batch, device=device)
            outputs = model(x, target["channel_mask"], target["time_mask"])
            arrays = {key: tensor.float().cpu().numpy() for key, tensor in outputs.items()}
        for i, item in enumerate(batch):
            n = len(item.grid)
            start, end, occupancy = [
                arrays[key][i, :n].astype(np.float64)
                for key in ("start_logits", "end_logits", "occupancy_logits")
            ]
            boxes, scores, _ = decode_boundaries(start, end, occupancy, arrays["presence_logits"][i], n)
            y = item.signal
            light, heavy = (y / np.maximum(y.max(axis=-1, keepdims=True), 1e-12)).mean(axis=1)
            curves = (
                light,
                heavy,
                np.exp(-np.logaddexp(0.0, -occupancy)),
                probability(start),
                probability(end),
            )
            candidates = []
            for box, score in zip(boxes, scores):
                lo, hi = np.interp(box, np.arange(n), item.grid)
                candidates.append((float(lo), float(hi), float(score)))
            yield item, curves, candidates


def predict(model, records, refiner, batch_size, device):
    center, scale, coef = [
        np.asarray(refiner[k])
        for k in ("standardization_mean", "standardization_scale", "coefficients")
    ]
    bias, cap = refiner["intercept"], refiner["max_width_fraction"]
    for item, curves, candidates in predict_raw(model, records, batch_size, device):
        adjusted = []
        for start, end, score in candidates:
            features = np.asarray(feature_pair(item.grid, curves, (start, end)))
            shifts = np.clip(((features - center) / scale) @ coef + bias, -cap, cap)
            low, high = np.clip(
                [start + shifts[0] * (end - start), end - shifts[1] * (end - start)],
                item.grid[0],
                item.grid[-1],
            )
            adjusted.append((float(low), float(high), score))
        yield item.group_id, adjusted


def save_predictions(path, predictions):
    opener = gzip.open if str(path).endswith(".gz") else open
    with (
        atomic_path(path) as temporary,
        opener(temporary, "wt", encoding="utf-8", newline="") as stream,
    ):
        writer = csv.writer(stream)
        writer.writerow(("group_id", "rank", "start_s", "end_s", "score"))
        for gid, candidates in predictions:
            if candidates:
                for rank, values in enumerate(candidates, 1):
                    writer.writerow((gid, rank, *values))
            else:
                writer.writerow((gid, 0, "", "", ""))


def load_predictions(path):
    groups = {}
    for row in csv_rows(path, ("group_id", "rank", "start_s", "end_s", "score")):
        gid, rank = row["group_id"], int(row["rank"])
        choices = groups.setdefault(gid, {})
        if not gid or rank in choices or not 0 <= rank <= 3:
            raise ValueError(f"Invalid candidate rank for {gid}")
        if rank == 0:
            if any(row[k] for k in ("start_s", "end_s", "score")):
                raise ValueError(f"Unexpected values at rank zero in {gid}")
            choices[rank] = None
        else:
            values = tuple(float(row[k]) for k in ("start_s", "end_s", "score"))
            if not np.isfinite(values).all() or values[0] >= values[1] or not 0 <= values[2] <= 1:
                raise ValueError(f"Invalid candidate in {gid}")
            choices[rank] = values
    for gid, choices in groups.items():
        if set(choices) != {0} and set(choices) != set(range(1, len(choices) + 1)):
            raise ValueError(f"Invalid ranks in {gid}")
    return {gid: choices.get(1) for gid, choices in groups.items()}
