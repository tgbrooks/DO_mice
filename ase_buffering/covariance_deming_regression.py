import numpy as np
import scipy.stats
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
        d = b2 - lam * b1
        S = V2 + lam**2 * V1 - lam * Cs
        w, U = np.linalg.eigh(S)
        k = w > w.max() * 1e-8  # generalized inverse
        z = U[:, k].T @ d
        return np.sum(z**2 / w[k])

    # First minimize Q over a discrete grid
    q = np.array([Q(lam) for lam in grid])
    i = np.argmin(q)
    # Then find the minimum by numeric search in the neighborhood of that grid point
    start_idx = max(i - 1, 0)
    end_idx = min(i + 1, len(grid) - 1)
    start = grid[start_idx]
    end = grid[end_idx]
    opt = scipy.optimize.minimize(Q, x0=grid[i], bounds=[(start, end)])
    # Profile confidence interval on lambda
    keep = q <= opt.fun + scipy.stats.chi2.ppf(0.95, df=1)
    if not any(keep):
        idx = np.arange(len(grid))
        keep = (idx >= start_idx) & (idx <= end_idx)
        assert any(keep)
    # test of lambda=1, chi2 test
    r = np.linalg.matrix_rank(V1 + V2)
    chi2_p = scipy.stats.chi2.sf(Q(1), df=r)
    return {
        "lambda": float(opt.x[0]),
        "Q": float(opt.fun),
        "chi2": float(Q(1)),
        "chi2_p": float(chi2_p),
        "df_gof": int(r - 1),
        "p_gof": float(scipy.stats.chi2.sf(opt.fun, df=r - 1)),
        "lo": float(min(grid[keep])),
        "hi": float(max(grid[keep])),
        "bounded": not (keep[0] or keep[len(keep) - 1]),
    }
