import pymc as pm
import pytensor.tensor as pt
import numpy as np
import scipy.linalg


def constrained_normal(
    name: str,
    orthog_to: np.ndarray,
    dims: tuple | None,
    sigma: float,
) -> pt.TensorVariable:
    """Forms a symmetric pm.Normal() variable that is constrained to be orthogonal to the row vectors of orthog_to
    Generalization of pm.ZeroSumNormal. Assumed 0 mean.

    Args:
        name: name of the resulting variable. Also creates {name}_raw lower-dimensional variables
            which will be samples to give the actual variable of interest.
        orthog_to: m x (n x ...) matrix
            Result will be orthogonal to the m dimensions spanned by the row vectors (or row matrices
            if orthog_to is more than 2 dimensional.
        dims: dimensions to use for the resulting value
        sigma: SD of the unconstrained, symmetric Normal.
    Returns:
        TensorVariable that is (n x  ...) shape, distributed as the uniform normal
        constrained to be orthogonal to the m vectors in orthog_to
    """

    out_shape = orthog_to.shape[1:]
    flattened = orthog_to.reshape((orthog_to.shape[0], -1))
    remainder = scipy.linalg.null_space(flattened)
    dim = remainder.shape[1]
    raw = pm.Normal(f"{name}_raw", sigma=sigma, shape=dim)
    return pm.Deterministic(
        name,
        (remainder @ raw).reshape(out_shape),
        dims=dims,
    )


def antisymmetric_constraints(n):
    """matrix of linear constraints to force an n x n matrix to be antisymmetric"""
    antisymmetric = np.zeros((n * n, n, n))
    for i in range(n):
        for j in range(n):
            # Note: some of these are redundant (gamma_ij = gamma_ji and gamma_ji = gamma_ij)
            # but they don't alter the end results since they don't change the nullspace
            antisymmetric[i + n * j, i, j] += 1
            antisymmetric[i + n * j, j, i] += 1
    return antisymmetric


def masked_constraints(mask):
    """matrix of linear constraints to force the masked values to be zero"""
    masked_cells = np.argwhere(mask == 0)
    c = np.zeros((len(masked_cells), *mask.shape))
    for r, (g, k) in enumerate(masked_cells):
        c[r, g, k] = 1.0
    return c
