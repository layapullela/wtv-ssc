"""Contiguous k-segment partitioning of an ordered affinity matrix via DP.
"""

import numpy as np
def dp_contiguous_ncut_partition(C, k, min_size=1, symmetrize=True):
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square (N, N) matrix.")
    N = C.shape[0]
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1.")
    if k > N:
        raise ValueError(f"k ({k}) cannot exceed N ({N}).")
    if min_size < 1:
        raise ValueError("min_size must be >= 1.")
    if k * min_size > N:
        raise ValueError(
            f"k * min_size ({k * min_size}) exceeds N ({N}); "
            "relax min_size or reduce k."
        )

    W = np.abs(C) + np.abs(C).T if symmetrize else np.array(C, copy=True)
    np.fill_diagonal(W, 0.0)  # no self-loops in within/degree accounting

    # 2D prefix sums for O(1) within(a, b) queries; S[i, j] = sum W[0:i, 0:j]
    S = np.zeros((N + 1, N + 1))
    S[1:, 1:] = np.cumsum(np.cumsum(W, axis=0), axis=1)
    deg = W.sum(axis=1)  # full-graph degree of each node
    deg_cum = np.concatenate([[0.0], np.cumsum(deg)])

    def within(a, b):
        return S[b, b] - S[a, b] - S[b, a] + S[a, a]

    def vol(a, b):
        return deg_cum[b] - deg_cum[a]

    # gain[a, b] = within(a,b) / vol(a,b) for segment [a, b); this is the
    # DP transition value (maximized <=> NCut minimized). vol == 0 (an
    # isolated segment) is given gain 0, matching the NCut convention that
    # an isolated cluster contributes the worst-case NCut_s = 1.
    NEG_INF = -np.inf
    gain = np.full((N + 1, N + 1), NEG_INF)
    for a in range(N):
        for b in range(a + min_size, N + 1):
            v = vol(a, b)
            gain[a, b] = within(a, b) / v if v > 0 else 0.0

    # dp[s, b] = best total gain using exactly s segments covering [0, b)
    dp = np.full((k + 1, N + 1), NEG_INF)
    dp[0, 0] = 0.0
    back = np.full((k + 1, N + 1), -1, dtype=int)

    for s in range(1, k + 1):
        lo = s * min_size
        hi = N - (k - s) * min_size
        for b in range(lo, hi + 1):
            a_lo = (s - 1) * min_size
            a_hi = b - min_size
            best_val, best_a = NEG_INF, -1
            for a in range(a_lo, a_hi + 1):
                prev = dp[s - 1, a]
                if prev == NEG_INF:
                    continue
                val = prev + gain[a, b]
                if val > best_val:
                    best_val, best_a = val, a
            dp[s, b] = best_val
            back[s, b] = best_a

    if dp[k, N] == NEG_INF:
        raise RuntimeError(
            "No feasible contiguous partition found; check k and min_size "
            "against N."
        )

    # Backtrack to recover boundaries.
    boundaries = [N]
    b, s = N, k
    while s > 0:
        a = back[s, b]
        boundaries.append(a)
        b, s = a, s - 1
    boundaries.reverse()

    labels = np.empty(N, dtype=int)
    for seg_idx in range(k):
        labels[boundaries[seg_idx]:boundaries[seg_idx + 1]] = seg_idx

    ncut_value = k - dp[k, N]
    return labels, boundaries, float(ncut_value)


def dp_contiguous_ncut_all_k(C, k_max, min_size=1, symmetrize=True):
    """DP-NCut for every feasible k in 1..k_max from one fill of the DP table.

    Unlike ``dp_contiguous_ncut_partition``, this does not prune states by the
    number of remaining segments, so ``dp[s, N]`` is populated for each s.
    Returns ``{k: (labels, boundaries, ncut_value)}``.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square (N, N) matrix.")
    N = C.shape[0]
    k_max = int(k_max)
    if min_size < 1:
        raise ValueError("min_size must be >= 1.")
    k_max = min(k_max, N // min_size)
    if k_max < 1:
        raise ValueError(
            f"k_max * min_size ({k_max * min_size}) exceeds N ({N}); "
            "relax min_size or reduce k_max."
        )

    W = np.abs(C) + np.abs(C).T if symmetrize else np.array(C, copy=True)
    np.fill_diagonal(W, 0.0)

    S = np.zeros((N + 1, N + 1))
    S[1:, 1:] = np.cumsum(np.cumsum(W, axis=0), axis=1)
    deg = W.sum(axis=1)
    deg_cum = np.concatenate([[0.0], np.cumsum(deg)])

    def within(a, b):
        return S[b, b] - S[a, b] - S[b, a] + S[a, a]

    def vol(a, b):
        return deg_cum[b] - deg_cum[a]

    NEG_INF = -np.inf
    gain = np.full((N + 1, N + 1), NEG_INF)
    for a in range(N):
        for b in range(a + min_size, N + 1):
            v = vol(a, b)
            gain[a, b] = within(a, b) / v if v > 0 else 0.0

    dp = np.full((k_max + 1, N + 1), NEG_INF)
    dp[0, 0] = 0.0
    back = np.full((k_max + 1, N + 1), -1, dtype=int)

    for s in range(1, k_max + 1):
        lo = s * min_size
        for b in range(lo, N + 1):
            a_lo = (s - 1) * min_size
            a_hi = b - min_size
            best_val, best_a = NEG_INF, -1
            for a in range(a_lo, a_hi + 1):
                prev = dp[s - 1, a]
                if prev == NEG_INF:
                    continue
                val = prev + gain[a, b]
                if val > best_val:
                    best_val, best_a = val, a
            dp[s, b] = best_val
            back[s, b] = best_a

    out = {}
    for k in range(1, k_max + 1):
        if dp[k, N] == NEG_INF:
            continue
        boundaries = [N]
        b, s = N, k
        while s > 0:
            a = back[s, b]
            boundaries.append(a)
            b, s = a, s - 1
        boundaries.reverse()
        labels = np.empty(N, dtype=int)
        for seg_idx in range(k):
            labels[boundaries[seg_idx]:boundaries[seg_idx + 1]] = seg_idx
        out[k] = (labels, boundaries, float(k - dp[k, N]))
    if not out:
        raise RuntimeError(
            "No feasible contiguous partition found; check k_max and min_size "
            "against N."
        )
    return out


def cluster_from_C_ordered(C, k, min_size=1, symmetrize=True):
    labels, _, _ = dp_contiguous_ncut_partition(
        C, k, min_size=min_size, symmetrize=symmetrize
    )
    return labels


if __name__ == "__main__":
    # Minimal sanity check on a synthetic 3-block affinity matrix.
    rng = np.random.default_rng(0)
    sizes = [50, 60, 90]
    labels_true = np.repeat(np.arange(3), sizes)
    N = len(labels_true)
    same = labels_true[:, None] == labels_true[None, :]
    C_syn = np.where(same, rng.uniform(0.4, 1.0, (N, N)),
                      rng.uniform(0.0, 0.05, (N, N)))
    np.fill_diagonal(C_syn, 0.0)

    labels, boundaries, ncut = dp_contiguous_ncut_partition(C_syn, k=3)
    from sklearn.metrics import adjusted_rand_score
    print("boundaries:", boundaries)
    print("NCut:", ncut)
    print("ARI vs ground truth:", adjusted_rand_score(labels_true, labels))