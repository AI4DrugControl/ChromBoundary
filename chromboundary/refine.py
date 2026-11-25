import numpy as np

OFFSETS = np.array([-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20])
CAP = 0.10


def probability(logits):
    x = np.exp(logits - np.max(logits))
    return x / x.sum()


def feature_pair(grid, curves, interval):
    start, end = interval[:2]
    width = end - start
    features = []
    for is_end, (edge, sign) in enumerate(((start, 1), (end, -1))):
        positions = edge + sign * width * OFFSETS

        samples = [np.interp(positions, grid, c) for c in (*curves[:3], curves[3 + is_end])]
        features.append(np.concatenate(([np.log(width), is_end], *samples)))
    return features
