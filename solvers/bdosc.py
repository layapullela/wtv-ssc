"""
BD-QOSC implementation, translated to python from BD_QOSC.m

NOTE ON TRANSLATION: several things in the original MATLAB are unconventional
or ambiguous; see the inline notes marked with "TRANSLATION NOTE" below.
"""

# translation of matlab code at: https://github.com/jbgao/BPFM

import numpy as np
from scipy.linalg import solve_sylvester

from dpcuts import cluster_from_C


def solve_l1l2(W, lambda_1):  # soft thresholding for columns
    m, n = np.shape(W)
    E = W
    for i in range(n):
        norm_col = np.linalg.norm(W[:, i])

        if norm_col > lambda_1:
            E[:, i] = (norm_col - lambda_1) * W[:, i] / norm_col
        else:
            E[:, i] = 0.0
    return E


def norm_l1(x):
    return np.sum(np.abs(x))


def norm_l1l2(x):
    L = 0
    for i in range(np.shape(x)[1]):
        L += np.linalg.norm(x[:, i])
    return L


def proj_kappa(Z0, S, k):

    # NOTE: assign_labels='discretize' uses yi-shi spectral discretization procedure, 
    # which is python equivalent to NcutDiscrete in Ncut from matlab. Close enough 
    # but the eigensolver, numerical implementation might be slightly different.

    from sklearn.cluster import SpectralClustering

    Z = np.zeros_like(Z0)
    temp = Z0 + S
    W = 0.5 * (np.abs(temp) + np.abs(temp.T))

    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=0,
    )
    labels = sc.fit_predict(W)

    for i in range(k):
        index = np.where(labels == i)[0]
        if index.size == 0:
            continue
        Z[np.ix_(index, index)] = Z0[np.ix_(index, index)]

    return Z


def bd_qosc(X, k, lambda_1, lambda_2, gamma_1, p, max_iter=200,
            diagconstraint=False, pos=False):
    """
    Problem
    -------
    min L(Z) = 1/2 ||X - XZ||^2_F + lambda_1 * ||Z^T Z||_1 + lambda_2 * ||ZR||_2/1

    where ||B||_2/1 = ||b_1||_2 + ||b_2||_2 + ... + ||b_n||_2

    Parameters
    ----------
    X : array (xm, xn)
        Data matrix.
    k : int
        Number of clusters used by `proj_kappa`'s Normalized Cut step, which
        is applied in the second half of the iterations to hard-project Z
        onto a k-block-diagonal support (see `proj_kappa` docstring).
    lambda_1, lambda_2 : float
        Regularization weights.
    gamma_1 : float
        Initial ADMM penalty parameter.
    p : float
        Multiplicative growth factor applied to gamma_1 every iteration.
        This is step (4) in their implementation, gamma_1 may blow up, something
        to be careful with.
    max_iter : int
        Maximum number of iterations.
    diagconstraint : bool
        If True, zero the diagonal of Z after each update.
    pos : bool
        If True, clip Z to be nonnegative after each update.

    Returns
    -------
    Z : array (xn, xn)
    func_vals : array
        Objective value per completed iteration.
    n_iterations : int
        Number of iterations actually run before stopping.
    """

    gamma_max = 10
    X = np.asarray(X, dtype=float)

    max_iterations = int(max_iter)
    func_vals = np.zeros(max_iterations)

    xm, xn = X.shape

    F = np.zeros((xn, xn - 1))
    Z = np.zeros((xn, xn))

    ones_R = np.ones((xn, xn - 1))
    R = (
        np.triu(ones_R, k=1) - np.triu(ones_R)
        + np.triu(ones_R, k=-1) - np.triu(ones_R)
    )

    U = Z @ R  # zeros, same shape as Z @ R

    XtX = X.T @ X
    RRt = R @ R.T

    n_iterations = max_iterations

    for it in range(max_iterations):

        # Step 1: Z-update via Sylvester equation
        # MATLAB: Z = lyap(A, B, C) solves A*Z + Z*B + C = 0, i.e. A*Z + Z*B = -C
        A = XtX
        B = 2.0 * lambda_1 * np.ones_like(Z) + gamma_1 * RRt
        Q = XtX + gamma_1 * (U @ R.T) + F @ R.T  # = -C
        Z = solve_sylvester(A, B, Q)

        if pos:
            Z[Z < 0] = 0.0

        if diagconstraint:
            np.fill_diagonal(Z, 0.0)

        # TRANSLATION NOTE: MATLAB check is `iteration > maxIterations/2`
        # with iteration 1-indexed; `it` here is 0-indexed, so `it + 1` is
        # the MATLAB iteration counter.
        if (it + 1) > max_iterations / 2.0:
            Z = proj_kappa(Z, Z, k)

        # Step 2: U-update (column-wise group soft thresholding)
        V = Z @ R - (1.0 / gamma_1) * F
        U = solve_l1l2(V, lambda_2 / gamma_1)

        # Step 3: dual update
        F = F + gamma_1 * (U - Z @ R)

        # Step 4: unconditional geometric growth of gamma_1 (see docstring note)
        gamma_1 = min(gamma_max, p * gamma_1)
        #gamma_1 = p * gamma_1

        # Objective
        func_vals[it] = (
            0.5 * np.linalg.norm(X - X @ Z, "fro") ** 2
            + lambda_1 * norm_l1(Z.T @ Z)
            + lambda_2 * norm_l1l2(Z @ R)
        )

        # Convergence checks (kept faithful to MATLAB's absolute thresholds,
        # unlike osc_exact's relative-residual tolerances)
        if it > 0:
            if func_vals[it] < 1e-3:
                n_iterations = it + 1
                break

        if it > 99:
            if (
                func_vals[it] < 1e-3
                or func_vals[it - 1] == func_vals[it]
                or func_vals[it - 1] - func_vals[it] < 1e-3
            ):
                n_iterations = it + 1
                break

    return Z, func_vals[:n_iterations], n_iterations


def cluster_from_Z(Z, k=None, k_max=None, method='eigengap', min_k=2, penalty=0.0):
    """Contiguous DP Normalized Cut on W = |Z| + |Z|^T."""
    return cluster_from_C(
        Z, k=k, k_max=k_max, method=method, min_k=min_k, penalty=penalty,
    )