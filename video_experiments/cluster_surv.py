"""Reproduce the surveillance test-set ARI scores recorded in
hyperparameters_surv.csv, by re-running OSC, TKSS, and WTV-SSC with the
exact hyperparameters that produced them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import SEED, N_FRAMES_HI, N_FRAMES_LO, apply_noise, reproduce  # noqa: E402

DATA_DIR = HERE.parent / "data"
SURVEILLANCE_ROOTS = (DATA_DIR / "surv_dataset" / "P1E",
                      DATA_DIR / "surv_dataset" / "P1L")
SURV_DOWN_HW = 24            # surveillance frames -> 24x24 -> n = 576
HYPERPARAMS_CSV = HERE / "hyperparameters_surv.csv"


def read_pgm(path):
    """Minimal binary (P5) PGM reader; the crops are 96x96 grayscale."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"{path}: expected P5 PGM, got {magic!r}")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        width, height = map(int, line.split())
        maxval = int(f.readline())
        dtype = np.uint8 if maxval < 256 else np.uint16
        pixels = np.frombuffer(f.read(), dtype=dtype)
    return pixels.reshape(height, width)


def downsample_image(img, size=SURV_DOWN_HW):
    pil = Image.fromarray(np.asarray(img, dtype=np.float32), mode="F")
    pil = pil.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float64)


def sequence_dirs():
    """The 24 sequence folders, P1E then P1L, each sorted by name."""
    seqs = []
    for root in SURVEILLANCE_ROOTS:
        seqs.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    return seqs


def load_sequence(root, rng):
    """One sequence -> (Y_raw, labels, person_dirs). Per person, n ~
    Unif{LO..HI} CONSECUTIVE frames from a random start."""
    lo, hi = N_FRAMES_LO, max(N_FRAMES_LO, N_FRAMES_HI)
    person_dirs = sorted((p for p in root.iterdir() if p.is_dir()),
                         key=lambda p: p.name)
    images, labels = [], []
    for person_idx, person_dir in enumerate(person_dirs):
        frames = sorted(person_dir.glob("*.pgm"), key=lambda p: p.name)
        n = min(int(rng.integers(lo, hi + 1)), len(frames))
        start = int(rng.integers(0, len(frames) - n + 1))
        for frame_path in frames[start:start + n]:
            images.append(downsample_image(read_pgm(frame_path)).reshape(-1))
            labels.append(person_idx)
    Y = np.stack(images, axis=1)
    return Y, np.asarray(labels, dtype=int), person_dirs


def load_all_sequences():
    """All 24 sequences, pixels scaled to [0,1]. ONE shared RNG stream."""
    rng = np.random.default_rng(SEED)
    loaded = []
    for seq in sequence_dirs():
        Y_raw, labels, person_dirs = load_sequence(seq, rng)
        loaded.append((seq.name, Y_raw / 255.0, labels, person_dirs))
        print(f"  loaded {seq.name}: Y={Y_raw.shape} people={len(person_dirs)}",
              flush=True)
    return loaded


def people_names(loaded):
    return sorted(p.name for p in loaded[0][3])


def chunk_folds(names, k, seed=SEED):
    """Shuffle the 25 people once, cut into disjoint groups of k."""
    names = np.asarray(sorted(names))
    rng = np.random.default_rng(seed)
    names = names[rng.permutation(len(names))]
    n_groups = len(names) // k
    names = names[: n_groups * k]
    return [sorted(g.tolist()) for g in names.reshape(n_groups, k)]


def subset_people(Y, labels, person_dirs, keep_names):
    """Keep only keep_names' columns, relabelled 0..k-1 in that order."""
    name_to_old = {p.name: i for i, p in enumerate(person_dirs)}
    ids = [name_to_old[n] for n in keep_names]
    mask = np.isin(labels, ids)
    remap = {int(old): new for new, old in enumerate(ids)}
    Ys = Y[:, mask]
    labs = np.array([remap[int(x)] for x in labels[mask]], dtype=int)
    return Ys, labs


def noisy_sequences(loaded, sigma):
    """One noisy Y per sequence for this sigma, shared across all folds."""
    rng = np.random.default_rng(SEED)
    return [(name, apply_noise(Y01, sigma, rng), labels, person_dirs)
            for name, Y01, labels, person_dirs in loaded]


def fold_test_matrices(loaded, k, sigma, fold):
    """The 24 test matrices (one per sequence) for one (k, sigma, fold)."""
    groups = chunk_folds(people_names(loaded), k)
    if not 0 <= fold < len(groups):
        raise ValueError(f"fold {fold} out of range: k={k} gives "
                         f"{len(groups)} folds")
    test_group = groups[fold]
    out = []
    for name, Y, labels, person_dirs in noisy_sequences(loaded, sigma):
        Ys, labs = subset_people(Y, labels, person_dirs, test_group)
        out.append((Ys, labs))
    return out


def main():
    print("loading all 24 sequences...", flush=True)
    loaded = load_all_sequences()
    reproduce(HYPERPARAMS_CSV,
              lambda k, sigma, fold: fold_test_matrices(loaded, k, sigma, fold))


if __name__ == "__main__":
    main()
