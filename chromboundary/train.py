import torch

from .data import atomic_path, collate, overlap, read_references, records
from .engine import batches, predict_raw
from .loss import training_loss
from .model import ChromBoundaryNet
from .refine import CAP, feature_pair


def fit_refiner(model, train, references, batch_size, device):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    features, targets = [], []
    for item, curves, choices in predict_raw(model, train, batch_size, device):
        bounds = references[item.group_id].bounds
        if bounds is None or not choices or overlap(choices[0], bounds) < 0.30:
            continue
        start, end, _ = choices[0]
        features.extend(feature_pair(item.grid, curves, (start, end)))
        targets.extend(((bounds[0] - start) / (end - start), (end - bounds[1]) / (end - start)))
    if not features:
        raise ValueError("No training positives matched at IoU 0.30 for endpoint refinement")
    scaler = StandardScaler().fit(features)
    ridge = Ridge(alpha=10.0).fit(scaler.transform(features), np.clip(targets, -CAP, CAP))
    return {
        "standardization_mean": scaler.mean_.tolist(),
        "standardization_scale": scaler.scale_.tolist(),
        "coefficients": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
        "max_width_fraction": CAP,
    }


def validation_key(model, validation, refs, profile, size, device):
    hits80 = hits90 = 0
    for record, _, choices in predict_raw(model, validation, size, device):
        bounds = refs[record.group_id].bounds
        if bounds is not None and choices:
            iou = overlap(bounds, choices[0])
            hits80 += int(iou >= 0.80)
            hits90 += int(iou >= 0.90)
    if profile == "metabolite":
        return hits90, hits80
    total = 0.0
    with torch.inference_mode():
        for batch in batches(validation, size):
            x, target = collate(batch, refs, device)
            outputs = model(x, target["channel_mask"], target["time_mask"])
            total += float(training_loss(outputs, target)) * len(batch)
    return hits80, -total / len(validation)


def train(signals, labels, profile, output, epochs, batch_size, seed, device, selection_panel=None):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epoch count and batch size must be positive")
    references = read_references(labels)
    train_ids = {k for k, r in references.items() if r.split == "train" and r.label >= 0}
    val_ids = {
        k
        for k, r in references.items()
        if r.split == "val" and r.label >= 0 and (selection_panel is None or r.panel == selection_panel)
    }
    for split, ids in (("train", train_ids), ("val", val_ids)):
        if not any(references[k].label == 1 for k in ids):
            raise ValueError(f"At least one labeled positive is required in {split}")
    training, validation = [], []
    for record in records(signals, profile, train_ids | val_ids):
        (training if record.group_id in train_ids else validation).append(record)
    torch.manual_seed(seed)
    model = ChromBoundaryNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    best_key, best_weights = None, None
    for epoch in range(1, epochs + 1):
        model.train()
        rng = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(training), generator=rng).tolist()
        total = 0.0
        for batch in batches((training[i] for i in order), batch_size):
            x, target = collate(batch, references, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, target["channel_mask"], target["time_mask"])
            loss = training_loss(outputs, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach().item() * len(batch)
        key = validation_key(model, validation, references, profile, batch_size, device)
        if best_key is None or key > best_key:
            best_key = key
            best_weights = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
        schedule.step()
        print(
            f"Epoch {epoch}/{epochs}  loss {total / len(training):.4f}  validation hits {key[0]}",
            flush=True,
        )
    model.load_state_dict(best_weights)
    refiner = fit_refiner(model, training, references, batch_size, device)
    with atomic_path(output) as temporary:
        torch.save(
            {"format_version": 1, "profile": profile, "model": best_weights, "refiner": refiner},
            temporary,
        )
