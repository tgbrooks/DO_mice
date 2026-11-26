"""
Paralog compensation model: does gene A's locus genotype affect its paralog B's
expression, in the opposite direction to its effect on A?

For an ordered pair (A = locus gene, B = response gene) on different chromosomes, a
negative binomial model of B's unique read counts (reads compatible with no other gene):

    y_j ~ NB(mu_j, dispersion)
    log mu_j = intercept + log(size factor_j) + sex + DO wave
               + log(exp(b_{h1}) + exp(b_{h2}))      B's own locus (h1, h2 = B diplotype)
               + (theta_{a1} + theta_{a2}) / 2       A's locus (a1, a2 = A diplotype)

b (B's cis effects) is a nuisance term; it also absorbs differences between B's
haplotypes in how many unique reads they produce. theta (sum to zero) is the effect of
A's locus on B. For a homozygous mouse, log B changes by theta_h; the total-counts model
gives log A changing by about beta_h, so theta = kappa * beta_A makes kappa the
elasticity of B with respect to A (kappa < 0: compensation).
"""

import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt

DEFAULTS = dict(
    sigma_beta=2.0,  # B's own haplotype effects
    sigma_theta=1.0,  # A-locus effects on B
    sigma_male=1.0,
    sigma_wave=0.5,
    target_accept=0.9,
)


def make_paralog_model(
    y: np.ndarray,
    hap_b1: np.ndarray,
    hap_b2: np.ndarray,
    hap_a1: np.ndarray,
    hap_a2: np.ndarray,
    male: np.ndarray,
    wave: np.ndarray,
    size_factor: np.ndarray,
    haplotypes: list[str],
    waves: list[str],
    mouse_ids: list[str],
    config: dict | None = None,
) -> pm.Model:
    """
    y: B's unique read counts per mouse. hap_*: haplotype indices (into haplotypes) of
    the B and A diplotypes. male: 0/1. wave: index into waves. size_factor: library size.
    """
    config = {**DEFAULTS, **(config or {})}
    coords = {"haplotypes": haplotypes, "waves": waves, "mice": mouse_ids}
    mean_expr = max(np.mean(y / size_factor), 1e-3) / 2
    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", mu=np.log(mean_expr), sigma=2)
        beta_b = pm.ZeroSumNormal("beta_b", sigma=config["sigma_beta"], dims="haplotypes")
        theta = pm.ZeroSumNormal("theta", sigma=config["sigma_theta"], dims="haplotypes")
        beta_male = pm.Normal("beta_male", sigma=config["sigma_male"])
        beta_wave = pm.ZeroSumNormal("beta_wave", sigma=config["sigma_wave"], dims="waves")
        log_disp = pm.Normal("log_disp", mu=np.log(0.05), sigma=2)
        log_mu = (
            intercept
            + np.log(size_factor)
            + pt.logaddexp(beta_b[hap_b1], beta_b[hap_b2])
            + (theta[hap_a1] + theta[hap_a2]) / 2
            + beta_male * male
            + beta_wave[wave]
        )
        pm.NegativeBinomial(
            "y", mu=pt.exp(log_mu), alpha=pt.exp(-log_disp), observed=y, dims="mice"
        )
    return model


def _convergence(post, var):
    sub = post[[var]]  # arviz 1.3 fails to find scalar variables by var_names
    rhat = np.atleast_1d(az.rhat(sub)[var].values)
    ess_bulk = np.atleast_1d(az.ess(sub, method="bulk")[var].values)
    ess_tail = np.atleast_1d(az.ess(sub, method="tail")[var].values)
    return dict(
        max_rhat=float(rhat.max()),
        min_ess_bulk=float(ess_bulk.min()),
        min_ess_tail=float(ess_tail.min()),
    )


def summarize_paralog_model(idata) -> dict:
    """theta and its posterior covariance (the results), plus nuisance terms and QC."""
    post = idata.posterior
    post = post.to_dataset() if hasattr(post, "to_dataset") else post
    stats = idata.sample_stats
    stats = stats.to_dataset() if hasattr(stats, "to_dataset") else stats
    obs = idata.observed_data
    obs = obs.to_dataset() if hasattr(obs, "to_dataset") else obs

    def draws(var):
        return post[var].stack(s=("chain", "draw")).transpose(..., "s").values

    theta = draws("theta")
    beta_b = draws("beta_b")
    return dict(
        theta=theta.mean(axis=1).tolist(),
        cov_theta=np.cov(theta).tolist(),
        beta_b=beta_b.mean(axis=1).tolist(),
        cov_beta_b=np.cov(beta_b).tolist(),
        beta_male=float(post["beta_male"].mean()),
        log_disp=float(post["log_disp"].mean()),
        diagnostics=dict(
            n_mice=int(obs["y"].size),
            n_chains=int(post.sizes["chain"]),
            n_draws=int(post.sizes["draw"]),
            n_divergences=int(stats["diverging"].sum()),
            theta=_convergence(post, "theta"),
            beta_b=_convergence(post, "beta_b"),
            log_disp=_convergence(post, "log_disp"),
        ),
    )
