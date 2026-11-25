import torch
from torch import nn
from torch.nn import functional as F


class PointwiseNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm1 = PointwiseNorm(width)
        self.norm2 = PointwiseNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        z = self.dropout(F.gelu(self.norm1(self.conv1(x)))) * valid
        z = self.dropout(self.norm2(self.conv2(z))) * valid
        return F.gelu(x + z) * valid


def _masked_pool(x, mask):
    count = mask.sum(-1).clamp_min(1.0)
    mean = (x * mask).sum(-1) / count
    peak = x.masked_fill(~mask.bool(), -1e4).amax(-1)
    return torch.cat((mean, peak), dim=1)


class ChromBoundaryNet(nn.Module):
    def __init__(self, width=32, dilations=(1, 2, 4, 8, 16, 32, 64), dropout=0.1):
        super().__init__()
        if width < 8 or not dilations:
            raise ValueError("width must be >=8 and dilations nonempty")

        self.stem = nn.Conv1d(4, width, 7, padding=3)
        self.stem_norm = PointwiseNorm(width)
        self.transition_block = TemporalBlock(width, 1, dropout)

        self.fuse = nn.Conv1d(2 * width, width, 1)
        self.fuse_norm = PointwiseNorm(width)
        self.temporal = nn.ModuleList([TemporalBlock(width, int(d), dropout) for d in dilations])
        self.dense_head = nn.Conv1d(width, 3, 1)
        self.presence_head = nn.Linear(2 * width, 1)

    def forward(self, x, channel_mask, time_mask):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("x must have shape B,2,C,T")
        batch, _, channels, length = x.shape
        if tuple(channel_mask.shape) != (batch, channels):
            raise ValueError("channel_mask must have shape B,C")
        if tuple(time_mask.shape) != (batch, length):
            raise ValueError("time_mask must have shape B,T")
        cm = channel_mask.bool()
        tm = time_mask.bool()
        if not bool(cm.any(1).all()) or not bool((tm.sum(1) >= 2).all()):
            raise ValueError("Every sample needs a transition and >=2 time points")
        if bool((tm[:, 1:] & ~tm[:, :-1]).any()):
            raise ValueError("time_mask must be a contiguous valid prefix")
        full_mask = cm[:, None, :, None] & tm[:, None, None, :]
        x = x * full_mask.to(x.dtype)

        scale = x.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        shape = x / scale
        inp = torch.cat((x, shape), dim=1).permute(0, 2, 1, 3)
        inp = inp.reshape(batch * channels, 4, length)
        tv = (cm[:, :, None] & tm[:, None, :]).reshape(batch * channels, 1, length).to(x.dtype)
        feats = F.gelu(self.stem_norm(self.stem(inp))) * tv
        feats = self.transition_block(feats, tv)
        count = cm.sum(dim=1, keepdim=True).to(feats.dtype)
        pair_weights = (cm.to(feats.dtype) / count)[:, :, None]
        pair_weights = pair_weights * tm[:, None, :].to(feats.dtype)
        feat = feats.reshape(batch, channels, -1, length)
        mean = (feat * pair_weights[:, :, None, :]).sum(1)
        variance = ((feat - mean[:, None, :, :]).square() * pair_weights[:, :, None, :]).sum(1)
        valid = tm[:, None, :].to(feats.dtype)
        z = F.gelu(self.fuse_norm(self.fuse(torch.cat((mean, variance), 1))))
        z = z * valid
        for block in self.temporal:
            z = block(z, valid)
        dense = self.dense_head(z)

        dense = dense.masked_fill(~tm[:, None, :], -1e4)
        presence = self.presence_head(_masked_pool(z, valid)).squeeze(1)
        return {
            "occupancy_logits": dense[:, 0],
            "start_logits": dense[:, 1],
            "end_logits": dense[:, 2],
            "presence_logits": presence,
        }


def endpoint_interval(outputs, time_mask):
    tm = time_mask.bool()
    start = outputs["start_logits"].float().masked_fill(~tm, -1e4)
    end = outputs["end_logits"].float().masked_fill(~tm, -1e4)
    ps = torch.softmax(start, -1) * tm.float()
    pe = torch.softmax(end, -1) * tm.float()
    left = ps.cumsum(-1)
    right = pe.flip(-1).cumsum(-1).flip(-1)
    node = left * right * tm.float()
    return node, ps, pe
