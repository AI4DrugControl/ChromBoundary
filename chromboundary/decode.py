import math
import numbers

import numpy as np


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    z = np.exp(-np.abs(values))
    return np.where(values >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max()
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _local_maxima(values, limit=32):
    if len(values) == 0:
        return np.empty(0, dtype=np.int64)
    beginnings = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1]
    endings = np.r_[beginnings[1:] - 1, len(values) - 1]
    centers = []
    for begin, end in zip(beginnings, endings):
        left_lower = begin == 0 or values[begin] > values[begin - 1]
        right_lower = end == len(values) - 1 or values[end] > values[end + 1]
        if left_lower and right_lower:
            centers.append((int(begin) + int(end)) // 2)
    centers = np.asarray(centers, dtype=np.int64)

    order = np.lexsort((centers, -values[centers]))
    return centers[order[:limit]]


def _empty_predictions():
    return (
        np.empty((0, 2), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.int64),
    )


def decode_boundaries(
    start_logits,
    end_logits,
    occupancy_logits,
    presence_logit,
    length,
    max_candidates=3,
    score_threshold=0.05,
    nms_threshold=0.3,
):
    if isinstance(length, bool) or not isinstance(length, numbers.Integral) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, numbers.Integral)
        or max_candidates < 0
    ):
        raise ValueError("max_candidates must be a nonnegative integer")
    if not math.isfinite(float(score_threshold)) or not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be finite and within [0, 1]")
    if not math.isfinite(float(nms_threshold)) or not 0 <= nms_threshold <= 1:
        raise ValueError("nms_threshold must be finite and within [0, 1]")
    arrays = []
    for name, values in (
        ("start_logits", start_logits),
        ("end_logits", end_logits),
        ("occupancy_logits", occupancy_logits),
    ):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or len(array) < length:
            raise ValueError(f"{name} must be 1D with at least length values")
        array = array[:length]
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains nonfinite valid time points")
        arrays.append(array)
    presence = np.asarray(presence_logit, dtype=np.float64)
    if presence.ndim != 0 or not np.isfinite(presence):
        raise ValueError("presence_logit must be one finite scalar")
    if length < 3 or max_candidates == 0:
        return _empty_predictions()
    p_present = float(_sigmoid(presence))
    if p_present < score_threshold:
        return _empty_predictions()

    start, end, occupancy = arrays
    p_start, p_end = _softmax(start), _softmax(end)
    log_p_occupancy = -np.logaddexp(0.0, -occupancy)
    starts, ends = _local_maxima(start), _local_maxima(end)
    start_cumulative = np.r_[0.0, np.cumsum(p_start)]
    end_cumulative = np.r_[0.0, np.cumsum(p_end)]
    occupancy_cumulative = np.r_[0.0, np.cumsum(log_p_occupancy)]

    candidates = []
    for s in starts:
        mass_s = start_cumulative[min(length, s + 3)] - start_cumulative[max(0, s - 2)]
        for e in ends:
            if e - s < 2:
                continue
            mass_e = end_cumulative[min(length, e + 3)] - end_cumulative[max(0, e - 2)]
            mean_log_inside = (occupancy_cumulative[e + 1] - occupancy_cumulative[s]) / (e - s + 1)
            inside = math.exp(min(0.0, float(mean_log_inside)))
            confidence = p_present * inside * math.sqrt(max(0.0, mass_s * mass_e))
            if confidence >= score_threshold:
                candidates.append((float(min(1.0, confidence)), int(s), int(e)))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    selected = []
    for confidence, start_index, end_index in candidates:
        suppressed = False
        for _, previous_start, previous_end in selected:
            overlap = max(0, min(end_index, previous_end) - max(start_index, previous_start))
            union = end_index - start_index + previous_end - previous_start - overlap
            if overlap / float(union) > nms_threshold:
                suppressed = True
                break
        if not suppressed:
            selected.append((confidence, start_index, end_index))
            if len(selected) == max_candidates:
                break
    if not selected:
        return _empty_predictions()
    boxes = np.asarray([[s, e] for _, s, e in selected], dtype=np.float32)
    scores = np.asarray([score for score, _, _ in selected], dtype=np.float32)
    return boxes, scores, np.ones(len(selected), dtype=np.int64)
