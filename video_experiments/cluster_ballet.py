"""Reproduce the ballet test-set ARI scores recorded in
hyperparameters_ballet.csv, by re-running OSC, TKSS, and WTV-SSC with the
exact hyperparameters that produced them.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import N_FRAMES_HI, N_FRAMES_LO, SEED, apply_noise, reproduce  # noqa: E402

DATA_DIR = HERE.parent / "data"
BALLET_FRAMES_DIR = DATA_DIR / "ballet_dataset" / "frames_tracked"
BALLET_LABELS_PATH = DATA_DIR / "ballet_dataset" / "labels.mat"
BALLET_DOWN_HW = 30                  # ballet frames -> 30x30 -> n = 900
BALLET_EXCLUDED_DANCERS = {8}        # these are 14 short cameo segments, an outlier sequence
BALLET_MATRICES_PER_DANCER = 12
HYPERPARAMS_CSV = HERE / "hyperparameters_ballet.csv"


def load_image(path, size=BALLET_DOWN_HW):
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return (np.array(img, dtype=np.float64) / 255.0).reshape(-1)


def sequence_dirs():
    return sorted(p for p in BALLET_FRAMES_DIR.iterdir() if p.is_dir())


def load_dancer_segments():
    """dancer_id -> [(seq_name, frame_start, frame_end)], frame_end exclusive."""
    seq_dirs = sequence_dirs()
    labels_by_seq = sio.loadmat(str(BALLET_LABELS_PATH))["labels"][0]
    if len(labels_by_seq) != len(seq_dirs):
        raise ValueError(f"labels.mat has {len(labels_by_seq)} sequences, "
                         f"found {len(seq_dirs)} in {BALLET_FRAMES_DIR}")
    segments = defaultdict(list)
    for seq_dir, arr in zip(seq_dirs, labels_by_seq):
        arr = arr.flatten()
        cur, start = int(arr[0]), 0
        for j in range(1, len(arr) + 1):
            if j == len(arr) or int(arr[j]) != cur:
                segments[cur].append((seq_dir.name, start, j))
                if j < len(arr):
                    cur, start = int(arr[j]), j
    return dict(segments)


def sample_combos(pool, k, n_combos, rng):
    """Sample n_combos distinct k-subsets of pool (as sorted lists)."""
    pool = list(pool)
    n_combos = min(int(n_combos), comb(len(pool), k))
    seen, groups, attempts = set(), [], 0
    max_attempts = max(n_combos * 1000, 1000)
    while len(groups) < n_combos:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"could not sample {n_combos} distinct "
                               f"{k}-subsets from {len(pool)} items")
        g = tuple(sorted(pool[i] for i in rng.choice(len(pool), size=k,
                                                      replace=False)))
        if g in seen:
            continue
        seen.add(g)
        groups.append(list(g))
    return groups


def load_segment_frames(seq_dir, lo, hi, rng):
    """n ~ Unif{LO..HI} CONSECUTIVE frames from a random start inside [lo, hi)."""
    frames = sorted(seq_dir.glob("*.jpg"))[lo:hi]
    n = min(int(rng.integers(N_FRAMES_LO, max(N_FRAMES_LO, N_FRAMES_HI) + 1)),
            len(frames))
    start = int(rng.integers(0, len(frames) - n + 1))
    return np.stack([load_image(p) for p in frames[start:start + n]], axis=1)


def build_pose_matrices(segments, k, n_matrices, name_to_dir, rng):
    """Up to n_matrices matrices pooling k of this dancer's pose-segments.

    Columns are concatenated in sorted segment-index order, so ground truth
    is k contiguous blocks.
    """
    n_all = comb(len(segments), k)
    combos = sample_combos(range(len(segments)), k, min(n_matrices, n_all), rng)
    out = []
    for i in range(n_matrices):
        combo = combos[i % len(combos)]
        cols, labs = [], []
        for pos, seg_idx in enumerate(combo):
            seq_name, lo, hi = segments[seg_idx]
            Yi = load_segment_frames(name_to_dir[seq_name], lo, hi, rng)
            cols.append(Yi)
            labs.append(np.full(Yi.shape[1], pos, dtype=int))
        out.append((np.concatenate(cols, axis=1), np.concatenate(labs)))
    return out


def eligible_dancers(dancer_segments, k):
    """Dancers with >= k segments, excluding dancer 8."""
    return sorted(d for d, segs in dancer_segments.items()
                  if len(segs) >= k and d not in BALLET_EXCLUDED_DANCERS)


def fold_test_matrices(dancer_segments, name_to_dir, k, sigma, fold):
    """The held-out dancer's noisy test matrices for leave-one-dancer-out fold."""
    eligible = eligible_dancers(dancer_segments, k)
    if not 0 <= fold < len(eligible):
        raise ValueError(f"fold {fold} out of range: k={k} gives "
                         f"{len(eligible)} folds (dancers {eligible})")
    held_out = eligible[fold]
    frame_rng = np.random.default_rng(SEED + held_out)
    clean = build_pose_matrices(dancer_segments[held_out], k,
                                 BALLET_MATRICES_PER_DANCER, name_to_dir, frame_rng)
    noise_rng = np.random.default_rng([SEED, held_out])
    return [(apply_noise(Y01, sigma, noise_rng), labels) for Y01, labels in clean]


def main():
    dancer_segments = load_dancer_segments()
    name_to_dir = {p.name: p for p in sequence_dirs()}
    reproduce(HYPERPARAMS_CSV,
              lambda k, sigma, fold: fold_test_matrices(
                  dancer_segments, name_to_dir, k, sigma, fold))


if __name__ == "__main__":
    main()
