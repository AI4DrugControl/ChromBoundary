from .data import overlap, read_references
from .engine import load_predictions


def evaluate(predictions, references, split=None):
    refs = read_references(references)
    calls = load_predictions(predictions)
    selected = {k: r for k, r in refs.items() if split is None or r.split == split}
    if not selected:
        raise ValueError("No reference groups match the selected split")
    if calls.keys() - refs.keys():
        raise ValueError("Predictions contain groups absent from the reference file")
    if selected.keys() - calls.keys():
        raise ValueError(
            "A prediction row is required for each evaluated group, including rank zero for no call"
        )
    positive = [k for k, ref in selected.items() if ref.label == 1]
    negative = [k for k, ref in selected.items() if ref.label == 0]
    ious = [overlap(selected[k].bounds, calls[k]) if calls[k] else 0.0 for k in positive]
    result = {
        "positive": len(positive),
        "negative": len(negative),
        "unknown": len(selected) - len(positive) - len(negative),
    }
    for threshold in (0.30, 0.80, 0.90):
        hits = sum(v >= threshold for v in ious)
        result[f"AR1@{threshold:.2f}"] = {
            "hits": hits,
            "recall": hits / len(positive) if positive else None,
        }
    result["negative_calls"] = sum(calls[k] is not None for k in negative)
    return result
