import numpy as np
import scipy.optimize


def covariance_deming_regression(b1, b2, V1, V2, grid=np.linspace(-2, 3, 2001)):
    """
    perform Deming-like regression when we have
    two k-vectors beta_1 and beta_2 each with known covariance matrices V_1 and V_2
    We assume no cross-covariance between beta_1 and beta_2
    Model: beta_1 = theta + epsilon_1
           beta_2 = lambda * theta + epsilon_2
           epsilon_i ~ N(0, V_i)
    So theta is the underlying 'true' parameter vector and lambda is the scaling between the two

    Q(lambda) := d(lambda)^T S(lambda)^-1 d(lambda)
      d(lambda) := beta_2 - lambda beta_1
      S(lambda) := V_2 + lambda^2 V_1
    so Q(lambda) is profile deviance (theta is profiled out) given covariances.
    Our solution is then the value of lambda that minimizes Q.

    Initially written by Claude Opus 5.
    """
    p = len(b1)
    assert len(b1) == len(b2)
    # Cross-covariance between betas, assumed to be zero
    C = np.zeros((p, p))
    Cs = C + C.T

    def Q(lam):
        # objective function to be minimized: -2 log likelihood (plus a constant)
        d = b2 - lam * b1
        S = V2 + lam**2 * V1 - lam * Cs
        e = np.linalg.eigh(S)
        k = e.eigenvalues > np.max(e.eigenvalues) * 1e-8  # generalized inverse
        return np.sum(
            d[k].T
            @ e.eigenvectors[:, k].T
            @ e.eigenvectors[:, k]
            @ d[k]
            / e.eigenvalues[k]
        )

    # First minimize Q over a discrete grid
    q = np.array([Q(lam) for lam in grid])
    i = np.argmin(q)
    # Then find the minimum by numeric search in the neighborhood of that grid point
    start = grid[max(i - 1, 0)]
    end = grid[min(i + 1, len(grid) - 1)]
    opt = scipy.optimize.minimize(Q, x0=grid[i], bounds=[(start, end)])
    # Profile confidence interval on lambda
    keep = q <= opt.fun + scipy.stats.chi2.ppf(0.95, df=1)
    # p-value testing whether lambda = 1, likelihood ratio test
    val_at_1 = Q(1)
    val_at_MLE = opt.fun
    lrt_p = scipy.stats.chi2.sf(val_at_1 - val_at_MLE, df=1)
    r = np.linalg.matrix_rank(V1 + V2)
    return {
        "lambda": opt.x[0],
        "Q": opt.fun,
        "lrt_p": lrt_p,
        "df_gof": r - 1,
        "p_gof": scipy.stats.chi2.sf(opt.fun, df=r - 1),
        "lo": min(grid[keep]),
        "hi": max(grid[keep]),
        "bounded": not (keep[0] or keep[len(keep) - 1]),
    }
