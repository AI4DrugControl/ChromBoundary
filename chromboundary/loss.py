import torch
from torch.nn import functional as F

from .model import endpoint_interval


def training_loss(outputs, targets):
    valid = targets["time_mask"].float()
    manual = targets["mask"].float() * valid
    logits = outputs["occupancy_logits"].float()
    peak = targets["has_peak"].float()
    positive = peak > 0.5
    count = valid.sum(-1).clamp_min(1.0)
    foreground = manual.sum(-1)
    pos_weight = ((count - foreground) / foreground.clamp_min(1.0)).clamp(1.0, 30.0)[:, None]
    weights = 1.0 + (pos_weight - 1.0) * manual
    error = F.binary_cross_entropy_with_logits(logits, manual, reduction="none")
    region_bce = ((error * weights * valid).sum(-1) / (weights * valid).sum(-1).clamp_min(1.0)).mean()
    prob = torch.sigmoid(logits) * valid
    dice = (1.0 - (2.0 * (prob * manual).sum(-1) + 1.0) / (prob.sum(-1) + foreground + 1.0)).mean()
    present = F.binary_cross_entropy_with_logits(outputs["presence_logits"].float(), peak)
    total = region_bce + dice + 0.2 * present
    if bool(positive.any()):
        tm = valid.bool()
        grid = torch.arange(logits.shape[1], dtype=torch.float32, device=logits.device)[None, :]
        node, ps, pe = endpoint_interval(outputs, tm)
        cross_entropies = []
        for key, value in (("start_logits", targets["starts"]), ("end_logits", targets["ends"])):
            centers = torch.where(positive, value.float(), torch.zeros_like(peak))
            soft = torch.exp(-0.5 * ((grid - centers[:, None]) / 1.5).square()) * valid
            soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-8)
            logp = F.log_softmax(outputs[key].float().masked_fill(~tm, -1e4), -1)
            cross_entropies.append(-(soft * logp).sum(-1)[positive].mean())
        endpoint = sum(cross_entropies) / 2.0
        order = (F.relu((ps * grid).sum(-1) - (pe * grid).sum(-1) + 1.0) / count)[positive].mean()
        consistency = (((prob - node).abs() * valid).sum(-1) / count)[positive].mean()
        total = region_bce + dice + 0.5 * endpoint + 0.2 * present + 0.1 * consistency + 0.1 * order
    return total
