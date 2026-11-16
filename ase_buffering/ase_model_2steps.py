"""
Two-stage allele-specific expression (ASE) model.

Stage 1  Per-mouse ASE (eta = logit p) with the read-class layer (q) and the fixed-
         dispersion negative-binomial likelihood. Each informative heterozygous mouse
         gets an independent, weakly informative prior.

Stage 2  Haplotype (beta), diplotype (gamma) and per-mouse spread (sigma_u) from a
         Gaussian approximation to each mouse's stage-1 *likelihood* for eta. The
         per-mouse eta are integrated out analytically:

             eta_hat ~ MvNormal(lin, C_lik + sigma_u^2 I)

Assumptions / caveats
  * The stage-1 prior on eta is divided out before stage 2, so it is not counted twice.
  * The Gaussian approximation is excellent for informative mice. It is poor when a
    mouse's likelihood is one-sided (near-monoallelic expression). `stage1_likelihood`
    reports per-mouse skewness so these cases can be flagged and, if needed, sent to
    the joint model instead.
  * Correlation between mice induced by shared q uncertainty is carried by the full
    stage-1 posterior covariance (stage2_cov="full"). "diag" ignores it.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import scipy.stats
import scipy.optimize

from util.compat_classes import CompatClassesDf

# Adjust this import to wherever these currently live.
from ase_buffering.ase_model import (
    estimate_class_proportions,
    estimate_class_alpha,
    eta_information,
)


DEFAULTS = dict(
    min_genotype_confidence=0.90,
    drop_all_zeros=True,
    # --- stage 1 ---
    sigma_q=0.5,
    eta_prior_sd=3.0,  # weak independent prior on each mouse's eta (divided out later)
    min_eta_info=1.0,  # mice with less plug-in information are not fit for eta
    min_n_informative=100,
    sigma_log_total_reads=1.0,
    # --- handoff ---
    cov_shrink=0.0,  # shrink stage-1 posterior correlations toward 0 (full mode)
    max_abs_skew=0.5,  # flag mice whose stage-1 posterior is clearly non-Gaussian
    # --- stage 2 ---
    gamma_mode="mvn",  # or "zero"
    sigma_beta=2.0,
    stage2_cov="full",  # "full" or "diag"
)


# ---------------------------------------------------------------------------
# Data preparation (shared by both stages)
# ---------------------------------------------------------------------------


@dataclass
class GeneData:
    ids: list
    counts: np.ndarray  # samples x classes
    hap1: np.ndarray
    hap2: np.ndarray
    eff_len1: np.ndarray  # effective length of hap1 per sample
    eff_len2: np.ndarray
    log_eff_ratio: np.ndarray  # log(eff_len1) - log(eff_len2), per mouse
    haplotypes: list
    class_names: list
    q_hat: np.ndarray  # haplotypes x classes
    alpha: np.ndarray  # samples x classes
    info: np.ndarray  # plug-in Fisher information about eta, per mouse
    est_hap_class_counts: np.ndarray  # haplotypes x classes
    pairs: list  # [(a, b), ...] with a < b
    pair_names: list
    idx: np.ndarray  # per mouse: index into pairs (0 for homozygotes)
    sign: np.ndarray  # per mouse: +1 / -1 for hets (AB vs BA), 0 for homozygotes
    Bm: np.ndarray  # pairs x haplotypes, rows e_a - e_b
    P_B: np.ndarray
    P_G: np.ndarray


def prepare_gene_data(
    gene_id: str,
    gene_class_counts: CompatClassesDf,
    diplotypes: pl.DataFrame,
    gene_effective_lengths: pl.DataFrame,
    config: dict,
) -> GeneData:
    total_counts = pl.DataFrame(
        {
            "mouse_id": gene_class_counts.ids,
            "total_counts": gene_class_counts.counts.sum(axis=1),
        }
    )

    gene_diplotypes = (
        diplotypes.filter(
            gene_id=gene_id,
        )
        .join(
            total_counts,
            "mouse_id",
        )
        .filter(
            # All-zero samples are useless for ASE model
            # (though they still need to be retained for the total counts model)
            pl.col("total_counts") > 0 if config.get("drop_all_zeros", True) else True,
            pl.col("genotype_confidence") > config["min_genotype_confidence"],
            pl.col("mouse_id").is_in(gene_class_counts.ids),
        )
        .select("diplotype", "mouse_id")
        .join(
            pl.DataFrame({"mouse_id": gene_class_counts.ids}),
            "mouse_id",
            how="left",
            maintain_order="right",
        )
        .with_columns(
            hap1=pl.col("diplotype").str.slice(0, 1),
            hap2=pl.col("diplotype").str.slice(1, 1),
        )
        .join(
            gene_effective_lengths.select(
                "mouse_id", "haplotype", eff_len1="effective_length"
            ),
            left_on=["mouse_id", "hap1"],
            right_on=["mouse_id", "haplotype"],
            maintain_order="left",
        )
        .join(
            gene_effective_lengths.select(
                "mouse_id", "haplotype", eff_len2="effective_length"
            ),
            left_on=["mouse_id", "hap2"],
            right_on=["mouse_id", "haplotype"],
            maintain_order="left",
        )
    )
    gene_class_counts = gene_class_counts.subset_samples(
        list(gene_diplotypes["mouse_id"])
    )
    assert list(gene_class_counts.ids) == gene_diplotypes["mouse_id"].to_list()

    haplotypes = list(gene_class_counts.haplotypes)
    hap_to_num = {h: i for i, h in enumerate(haplotypes)}
    hap1 = gene_diplotypes.select(pl.col("hap1").replace_strict(hap_to_num))[
        "hap1"
    ].to_numpy()
    hap2 = gene_diplotypes.select(pl.col("hap2").replace_strict(hap_to_num))[
        "hap2"
    ].to_numpy()
    eff_len1 = gene_diplotypes["eff_len1"].to_numpy()
    eff_len2 = gene_diplotypes["eff_len2"].to_numpy()
    log_eff_ratio = np.log(eff_len1) - np.log(eff_len2)

    counts = gene_class_counts.counts
    n_samples, n_classes = counts.shape
    n_haps = len(haplotypes)
    class_names = [
        "".join(h for j, h in enumerate(haplotypes) if gene_class_counts.compat[i, j])
        for i in range(n_classes)
    ]

    q_hat = estimate_class_proportions(counts, hap1, hap2, n_haps)
    # TODO: estimate_class_alpha fails if no diplotype has >= 4 mice; needs a fallback curve.
    alpha, _ = estimate_class_alpha(gene_class_counts, q_hat, hap1, hap2)
    info = eta_information(counts, q_hat, hap1, hap2, alpha)

    est_hap_total_counts = np.array(
        [counts[(hap1 == i) | (hap2 == i)].sum(axis=0) / 2 for i in range(n_haps)]
    )
    est_hap_class_counts = est_hap_total_counts * q_hat

    pairs = [(a, b) for a in range(n_haps) for b in range(a + 1, n_haps)]
    pair_index = {pr: k for k, pr in enumerate(pairs)}
    pair_names = [f"pair_{haplotypes[a]}{haplotypes[b]}" for a, b in pairs]
    Bm = np.zeros((len(pairs), n_haps))
    for k, (a, b) in enumerate(pairs):
        Bm[k, a], Bm[k, b] = 1.0, -1.0
    P_B = Bm @ Bm.T / n_haps
    P_G = np.eye(len(pairs)) - P_B

    idx = np.zeros(n_samples, dtype=int)
    sign = np.zeros(n_samples)
    for j, (a, b) in enumerate(zip(hap1, hap2)):
        if a != b:
            idx[j] = pair_index[(min(a, b), max(a, b))]
            sign[j] = 1.0 if a < b else -1.0

    return GeneData(
        ids=list(gene_class_counts.ids),
        counts=counts,
        hap1=hap1,
        hap2=hap2,
        eff_len1=eff_len1,
        eff_len2=eff_len2,
        log_eff_ratio=log_eff_ratio,
        haplotypes=haplotypes,
        class_names=class_names,
        q_hat=q_hat,
        alpha=alpha,
        info=info,
        est_hap_class_counts=est_hap_class_counts,
        pairs=pairs,
        pair_names=pair_names,
        idx=idx,
        sign=sign,
        Bm=Bm,
        P_B=P_B,
        P_G=P_G,
    )


# ---------------------------------------------------------------------------
# Stage 1: per-mouse eta + q + NB likelihood
# ---------------------------------------------------------------------------


def q_information(counts, q_hat, hap1, hap2, alpha, p=0.5):
    """Fisher information about each log-q entry (haplotypes x classes), NB likelihood."""
    T = counts.sum(axis=1, keepdims=True)
    mu = T * (p * q_hat[hap1] + (1 - p) * q_hat[hap2])
    var = mu * (1 + mu / alpha)
    info = np.zeros_like(q_hat)
    for h in range(q_hat.shape[0]):
        w = p * (hap1 == h) + (1 - p) * (hap2 == h)  # homozygotes of h get weight 1
        dmu = T * w[:, None] * q_hat[h] * (1 - q_hat[h])
        info[h] = (dmu**2 / var).sum(axis=0)
    return info


def _make_q(d: GeneData, config: dict):
    n_haps, n_classes = d.q_hat.shape
    if config.get("q_fixed") is not None:
        # Oracle / diagnostic mode: q is known, not sampled
        q_fixed = np.asarray(config["q_fixed"], dtype=float)
        assert q_fixed.shape == (n_haps, n_classes)
        return pm.Deterministic(
            "q", pt.as_tensor_variable(q_fixed), dims=("haplotypes", "classes")
        )
    log_q_hat = np.log(d.q_hat)

    # Pin the largest class of each haplotype (softmax invariance); tiny classes have
    # one-sided likelihoods, so they are never centered.
    pin_idx = np.argmax(d.est_hap_class_counts, axis=1)
    pin = np.arange(n_classes)[None, :] == pin_idx[:, None]

    q_logit_raw = pm.Normal(
        "q_logit_raw", sigma=config["sigma_q"], dims=("haplotypes", "classes")
    )
    q_logit = pt.switch(pin, 0.0, q_logit_raw)
    return pm.Deterministic(
        "q",
        pm.math.softmax(log_q_hat + q_logit, axis=1),
        dims=("haplotypes", "classes"),
    )


def _nb_potential(counts: np.ndarray, alpha: np.ndarray, mu):
    # Constant terms (fixed alpha, observed counts) dropped: they do not affect sampling.
    # ll = -alpha * pt.log1p(mu / alpha) + counts * (pt.log(mu) - pt.log(alpha + mu))
    ll = (
        pt.gammaln(counts + alpha)
        - pt.gammaln(alpha)
        - pt.gammaln(counts + 1)
        + alpha * (pt.log(alpha) - pt.log(alpha + mu))
        + counts * (pt.log(mu) - pt.log(alpha + mu))
    )
    pm.Potential("ll", pt.sum(ll))


def stage1_fit_mask(d: GeneData, config: dict) -> np.ndarray:
    """Mice whose eta is sampled in stage 1: informative heterozygotes."""
    return (d.sign != 0) & (d.info > config["min_eta_info"])


def make_stage1_model(d: GeneData, config: dict) -> pm.Model:
    n_samples = len(d.ids)
    i_fit = np.nonzero(stage1_fit_mask(d, config))[0]

    n_informative = len(i_fit)
    if n_informative < config["min_n_informative"]:
        raise ValueError(f"Only {n_informative} samples were informative for ASE")

    coords = {
        "samples": d.ids,
        "haplotypes": d.haplotypes,
        "classes": d.class_names,
        "fit_samples": [d.ids[i] for i in i_fit],
    }
    with pm.Model(coords=coords) as model:
        q = _make_q(d, config)

        # Independent, weak priors centered on the effective-length offset.
        eta_fit = pm.Normal(
            "eta",
            mu=d.log_eff_ratio[i_fit],
            sigma=config["eta_prior_sd"],
            dims="fit_samples",
        )
        # Homozygotes: p does not enter the likelihood. Uninformative hets: fixed at the
        # offset; their reads barely depend on p but still inform q.
        eta = pt.as_tensor(d.log_eff_ratio.astype(float))
        eta = pt.set_subtensor(eta[i_fit], eta_fit)
        p = pm.math.sigmoid(eta)

        class_props = p[:, None] * q[d.hap1] + (1 - p)[:, None] * q[d.hap2]
        observed_totals = d.counts.sum(axis=1)
        log_total_reads = pm.Normal(
            "log_total_reads",
            mu=np.log(observed_totals),
            sigma=config["sigma_log_total_reads"],
            dims="samples",
        )
        _nb_potential(d.counts, d.alpha, pt.exp(log_total_reads[:, None]) * class_props)

    model.i_fit = i_fit  # convenience for stage1_likelihood
    assert len(i_fit) <= n_samples
    return model


def stage1_likelihood(idata, d: GeneData, config: dict) -> dict:
    """
    Gaussian approximation to the stage-1 likelihood for eta (prior divided out).

    Returns dict with:
        index    mouse indices (into d.ids) used in stage 2
        mean     likelihood mean for eta, per mouse
        cov      likelihood covariance (diagonal if stage2_cov == "diag")
        skew     per-mouse posterior skewness (non-Gaussianity flag)
        flagged  mice with |skew| > max_abs_skew (left in; inspect or refit jointly)
    """
    i_fit = np.nonzero(stage1_fit_mask(d, config))[0]
    draws = (
        idata.posterior["eta"]
        .stack(s=("chain", "draw"))
        .transpose("fit_samples", "s")
        .values
    )  # (n_fit, S)

    post_mean = draws.mean(axis=1)
    post_cov = np.atleast_2d(np.cov(draws))
    skew = scipy.stats.skew(draws, axis=1)

    if config["stage2_cov"] == "diag":
        post_cov = np.diag(np.diag(post_cov))
    else:
        lam = config["cov_shrink"]
        post_cov = (1 - lam) * post_cov + lam * np.diag(np.diag(post_cov))

    # Divide out the stage-1 prior: Lambda_lik = Lambda_post - Lambda_prior
    tau2 = config["eta_prior_sd"] ** 2
    prior_mean = d.log_eff_ratio[i_fit]
    P_post = np.linalg.inv(post_cov)
    P_lik = P_post - np.eye(len(i_fit)) / tau2
    P_lik = 0.5 * (P_lik + P_lik.T)

    # Guard against Monte Carlo noise making P_lik indefinite. Clipped directions are
    # ones the data barely inform; giving them huge variance drops them harmlessly.
    w, V = np.linalg.eigh(P_lik)
    w = np.maximum(w, 1e-8)
    C_lik = (V / w) @ V.T
    lik_mean = C_lik @ (P_post @ post_mean - prior_mean / tau2)

    if config["stage2_cov"] == "diag":
        C_lik = np.diag(np.diag(C_lik))

    return dict(
        index=i_fit,
        mean=lik_mean,
        cov=C_lik,
        skew=skew,
        flagged=[
            d.ids[i] for i, s in zip(i_fit, skew) if abs(s) > config["max_abs_skew"]
        ],
    )


# ---------------------------------------------------------------------------
# Stage 2: beta / gamma / sigma_u with eta integrated out
# ---------------------------------------------------------------------------


def fit_stage2_gls(d, lik, sigma_u_max=5.0, tol=1e-8):
    """
    Extract beta values from the stage 1 eta fits and covariance by GLS

    Model:
     eta_hat - offsets = X Beta + epsilon
     epsilon ~ N(0, V(sigma_u^2))
     V(sigma_u^2) = C_lik + sigma_u^2 I
     where C_lik is from stage 1 and X has +1 at hap1 and -1 at hap2
     Estimate sigma_u^2 from REML and then beta from ML
    """

    i = lik["index"]
    y = lik["mean"] - d.log_eff_ratio[i]
    C = lik["cov"]
    n, n_h = len(i), len(d.haplotypes)

    X = np.zeros((n, n_h))
    X[np.arange(n), d.hap1[i]] += 1.0
    X[np.arange(n), d.hap2[i]] -= 1.0

    # Sum-to-zero beta: orthonormal basis Z, then keep only directions the data identify
    Z = scipy.linalg.null_space(np.ones((1, n_h)))  # n_h x (n_h - 1)
    U, s, Vt = np.linalg.svd(X @ Z, full_matrices=False)
    keep = s > tol * s.max()
    B = Z @ Vt[keep].T  # beta = B @ theta
    Xk = X @ B

    def solve(s2):
        L = scipy.linalg.cho_factor(C + s2 * np.eye(n))
        A = Xk.T @ scipy.linalg.cho_solve(L, Xk)
        theta = np.linalg.solve(A, Xk.T @ scipy.linalg.cho_solve(L, y))
        r = y - Xk @ theta
        logdetV = 2 * np.log(np.diag(L[0])).sum()
        nll = 0.5 * (
            logdetV + np.linalg.slogdet(A)[1] + r @ scipy.linalg.cho_solve(L, r)
        )
        return nll, theta, A

    res = scipy.optimize.minimize_scalar(
        lambda su: solve(su**2)[0], bounds=(0.0, sigma_u_max), method="bounded"
    )
    sigma_u = res.x
    _, theta, A = solve(sigma_u**2)
    beta = B @ theta
    cov_beta = B @ np.linalg.inv(A) @ B.T
    identified = (
        np.linalg.matrix_rank(B) == n_h - 1
    )  # False if some haplotype has no information
    return dict(beta=beta, cov_beta=cov_beta, sigma_u=sigma_u, identified=identified)


def fit_stage2_eb(d, lik, sigma_beta=2.0, bound=5.0):
    """
    Extract beta values via ridge regression

    Model:
        eta - offsets = X Delta + u
        Delta_gh = Beta_g - Beta_h + gamma_gh
        u ~ N(0, sigma_u)
        gamma_gh ~ N(0, sigma_gamma)
        Beta ~ N(0, sigma_beta)
        where X is the matrix of samples' diplotypes

    """
    i = lik["index"]
    y = lik["mean"] - d.log_eff_ratio[i]
    C = lik["cov"]
    n_obs, n_pairs, n_h = len(i), len(d.pairs), len(d.haplotypes)
    S = np.zeros((n_obs, n_pairs))
    S[np.arange(n_obs), d.idx[i]] = d.sign[i]

    def Sigma_delta(sg):
        return n_h * sigma_beta**2 * d.P_B + sg**2 * d.P_G

    def nll(params):
        su, sg = params
        M = S @ Sigma_delta(sg) @ S.T + C + su**2 * np.eye(n_obs)
        L = scipy.linalg.cho_factor(M)
        return np.log(np.diag(L[0])).sum() + 0.5 * y @ scipy.linalg.cho_solve(L, y)

    res = scipy.optimize.minimize(
        nll, x0=[0.1, 0.1], bounds=[(0, bound), (0, bound)], method="L-BFGS-B"
    )
    su, sg = res.x

    # Posterior of delta in the form that stays valid when sigma_gamma -> 0 (singular prior)
    Sd = Sigma_delta(sg)
    M = S @ Sd @ S.T + C + su**2 * np.eye(n_obs)
    L = scipy.linalg.cho_factor(M)
    K = Sd @ S.T  # n_pairs x n_obs
    delta_mean = K @ scipy.linalg.cho_solve(L, y)
    delta_cov = Sd - K @ scipy.linalg.cho_solve(L, K.T)

    T = d.Bm.T / n_h  # beta = T @ delta
    return dict(
        sigma_u=su,
        sigma_gamma=sg,
        beta=T @ delta_mean,
        cov_beta=T @ delta_cov @ T.T,
        gamma=d.P_G @ delta_mean,
        cov_gamma=d.P_G @ delta_cov @ d.P_G.T,
        delta=delta_mean,
        cov_delta=delta_cov,
        converged=res.success,
    )


# ---------------------------------------------------------------------------
# Sandwich variant: the noisy n x n stage-1 covariance is never inverted
# ---------------------------------------------------------------------------
#
# Notation (matches fit_stage2_eb):
#   S        n_obs x n_pairs sign/design matrix (eta = S delta + u + e)
#   Sigma    stage-1 posterior covariance of eta (estimated by np.cov of draws)
#   C        stage-1 *likelihood* covariance of eta, C = Sigma (I - Sigma/tau^2)^-1
#   R        C + sigma_u^2 I, the residual covariance of y around S delta
#
# fit_stage2_eb inverts M = S Sd S' + C_hat + sigma_u^2 I with a noisy C_hat. When C has
# shared-q correlation (aligned with the beta directions), the resulting cov_delta is
# too small, increasingly so with fewer draws (in synthetic tests ~1.5x at 400 draws,
# ~1.1-1.2x at 2000; fine without shared-q correlation). Here the weights W come from
# a working covariance built only from per-mouse variances (each estimated from all
# draws, so nearly noise-free), and the actual error variance W R W' uses the full C
# only through that small (n_pairs x n_pairs) quadratic form: C is never inverted, and
# the cross-mouse correlation from shared q is kept.


def stage1_draws(idata, d: GeneData, config: dict) -> dict:
    """Per-mouse stage-1 summaries for fit_stage2_sandwich (prior divided out).

    Returns dict with:
        index           mouse indices (into d.ids) used in stage 2
        mean            likelihood mean for eta, per mouse
        var_diag        likelihood variance for eta, per mouse (marginal only)
        cov             full likelihood covariance C (used only as W C W' in stage 2)
        tau2            stage-1 prior variance on eta
        skew, flagged   as in stage1_likelihood
    """
    i_fit = np.nonzero(stage1_fit_mask(d, config))[0]
    eta = idata.posterior["eta"]
    # Stage 2 recomputes which mice were fit; make sure that matches stage 1 exactly
    fit_ids = [str(x) for x in eta["fit_samples"].values]
    if fit_ids != [d.ids[i] for i in i_fit]:
        raise ValueError(
            "Stage-1 fit_samples do not match the mice selected by stage1_fit_mask "
            f"({len(fit_ids)} vs {len(i_fit)}); was stage 1 run with different data/config?"
        )
    draws = eta.stack(s=("chain", "draw")).transpose("fit_samples", "s").values
    return _stage1_draws_from_array(draws, d, i_fit, config)


def _stage1_draws_from_array(draws, d: GeneData, i_fit, config: dict) -> dict:
    n_draws = draws.shape[1]
    post_mean = draws.mean(axis=1)
    Dc = (draws - post_mean[:, None]) / np.sqrt(n_draws - 1)
    skew = scipy.stats.skew(draws, axis=1)

    tau2 = config["eta_prior_sd"] ** 2
    prior_mean = d.log_eff_ratio[i_fit]
    n_fit = len(i_fit)
    # Exact Gaussian prior division without inverting Sigma:
    #   C = (Sigma^-1 - I/tau^2)^-1 = Sigma (I - Sigma/tau^2)^-1
    #   mean_lik = C (Sigma^-1 m - m0/tau^2) = m + C (m - m0) / tau^2
    # (Sigma and I - Sigma/tau^2 commute; the latter has eigenvalues near 1.)
    Sigma = Dc @ Dc.T
    divisor = np.eye(n_fit) - Sigma / tau2
    C = np.linalg.solve(divisor, Sigma)
    C = 0.5 * (C + C.T)
    lik_mean = post_mean + C @ (post_mean - prior_mean) / tau2

    # Diagnostics: how strongly correlated are the mice's stage-1 etas?
    post_sd = np.sqrt(np.diag(Sigma))
    corr = Sigma / np.outer(post_sd, post_sd)
    off = corr[~np.eye(n_fit, dtype=bool)]

    return dict(
        index=i_fit,
        mean=lik_mean,
        var_diag=np.diag(C).copy(),
        cov=C,
        tau2=tau2,
        post_mean=post_mean,
        post_var=np.diag(Sigma).copy(),
        n_draws=n_draws,
        # Must be > 0 for the prior division to be valid; near 1 normally
        min_eig_prior_division=float(np.linalg.eigvalsh(divisor).min()),
        mean_abs_corr=float(np.abs(off).mean()) if off.size else 0.0,
        max_abs_corr=float(np.abs(off).max()) if off.size else 0.0,
        skew=skew,
        flagged=[
            d.ids[i] for i, s in zip(i_fit, skew) if abs(s) > config["max_abs_skew"]
        ],
    )


def fit_stage2_sandwich(d, lik, sigma_beta=2.0, bound=5.0):
    """
    Same model and point estimator as fit_stage2_eb, but with a working diagonal
    stage-1 covariance for the weights and a draw-based (sandwich) variance.

    Model:
        y = eta_lik - offsets = S delta + u + e
        delta ~ N(0, Sd),  Sd = n_h sigma_beta^2 P_B + sigma_gamma^2 P_G
        u ~ N(0, sigma_u^2 I),  e ~ N(0, C)

    Steps:
        1. sigma_u, sigma_gamma by ML with C replaced by C_w = diag(var_diag)
        2. delta_hat = W y,  W = Sd S' (S Sd S' + C_w + sigma_u^2 I)^-1
        3. cov(delta_hat - delta) = (W S - I) Sd (W S - I)' + W C W' + sigma_u^2 W W'
           with C the full stage-1 likelihood covariance from stage1_draws.
       If C really were C_w, step 3 equals the usual posterior covariance
       Sd - W S Sd (returned as cov_*_working, for comparison).
    """
    i = lik["index"]
    y = lik["mean"] - d.log_eff_ratio[i]
    c_w = lik["var_diag"]
    C = lik["cov"]
    n_obs, n_pairs, n_h = len(i), len(d.pairs), len(d.haplotypes)
    S = np.zeros((n_obs, n_pairs))
    S[np.arange(n_obs), d.idx[i]] = d.sign[i]

    def Sigma_delta(sg):
        return n_h * sigma_beta**2 * d.P_B + sg**2 * d.P_G

    def working_M(su, sg):
        return S @ Sigma_delta(sg) @ S.T + np.diag(c_w + su**2)

    def nll(params):
        L = scipy.linalg.cho_factor(working_M(*params))
        return np.log(np.diag(L[0])).sum() + 0.5 * y @ scipy.linalg.cho_solve(L, y)

    res = scipy.optimize.minimize(
        nll, x0=[0.1, 0.1], bounds=[(0, bound), (0, bound)], method="L-BFGS-B"
    )
    su, sg = res.x

    Sd = Sigma_delta(sg)
    L = scipy.linalg.cho_factor(working_M(su, sg))
    W = scipy.linalg.cho_solve(L, S @ Sd).T  # n_pairs x n_obs (M is symmetric)
    delta_mean = W @ y

    # Prior / shrinkage part (large along directions the data do not identify)
    A = W @ S - np.eye(n_pairs)
    prior_part = A @ Sd @ A.T
    # Stage-1 noise part: C enters only through this small quadratic form, never inverted
    noise_part = W @ C @ W.T
    delta_cov = prior_part + noise_part + su**2 * (W @ W.T)
    delta_cov = 0.5 * (delta_cov + delta_cov.T)
    delta_cov_working = Sd - W @ S @ Sd
    delta_cov_working = 0.5 * (delta_cov_working + delta_cov_working.T)

    T = d.Bm.T / n_h  # beta = T @ delta

    # Diagnostics
    def tr_beta(M):
        return float(np.trace(T @ M @ T.T))

    noise_part_diag = W @ np.diag(np.diag(C)) @ W.T
    resid = y - S @ delta_mean
    return dict(
        sigma_u=su,
        sigma_gamma=sg,
        beta=T @ delta_mean,
        cov_beta=T @ delta_cov @ T.T,
        gamma=d.P_G @ delta_mean,
        cov_gamma=d.P_G @ delta_cov @ d.P_G.T,
        delta=delta_mean,
        cov_delta=delta_cov,
        cov_beta_working=T @ delta_cov_working @ T.T,
        converged=res.success,
        diagnostics=dict(
            nll=float(res.fun),
            optimizer_message=str(res.message),
            n_obs=int(n_obs),
            # trace of cov_beta split by source (sums to tr(cov_beta))
            tr_cov_beta_prior_part=tr_beta(prior_part),
            tr_cov_beta_noise_part=tr_beta(noise_part),
            tr_cov_beta_sigma_u_part=tr_beta(su**2 * (W @ W.T)),
            # > 1 when cross-mouse correlation in C (e.g. shared q) adds variance
            noise_part_vs_diagonal_C=tr_beta(noise_part) / tr_beta(noise_part_diag),
            # sandwich vs. model-based (working, diagonal C) covariance of beta
            tr_cov_beta_over_working=tr_beta(delta_cov) / tr_beta(delta_cov_working),
            # ~1 if the working variances (C_w + sigma_u^2) fit the residuals
            mean_sq_std_resid=float(np.mean(resid**2 / (c_w + su**2))),
        ),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def fit_gene_two_stage(
    gene_id: str,
    gene_class_counts: CompatClassesDf,
    diplotypes: pl.DataFrame,
    gene_effective_lengths: pl.DataFrame,
    config: dict | None = None,
    stage1_sample_kwargs: dict | None = None,
):
    cfg = {**DEFAULTS, **(config or {})}
    d = prepare_gene_data(
        gene_id, gene_class_counts, diplotypes, gene_effective_lengths, cfg
    )

    with make_stage1_model(d, cfg):
        idata1 = pm.sample(**(stage1_sample_kwargs or {}))

    lik = stage1_likelihood(idata1, d, cfg)

    stage2 = fit_stage2_eb(d, lik)
    summary = summarize_ase_models(idata1, stage2, d)

    return dict(
        data=d,
        config=cfg,
        idata_stage1=idata1,
        likelihood=lik,
        stage2=stage2,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


def _summary(idata, **kwargs):
    return pl.DataFrame(az.summary(idata, **kwargs).reset_index(names="variable"))


def summarize_ase_models(idata_ase1, stage2, gene_data) -> dict:
    eta_summary = _summary(idata_ase1, var_names=["eta"])
    return {
        "beta": np.asarray(stage2["beta"]).tolist(),
        "gamma": np.asarray(stage2["gamma"]).tolist(),
        "delta": np.asarray(stage2["delta"]).tolist(),
        "eta": eta_summary.with_columns(
            mouse_id=pl.col("variable").str.extract(r"eta\[(.*)\]", 1)
        ).to_dicts(),
        "cov": {
            "cov_beta": np.asarray(stage2["cov_beta"]).tolist(),
            "cov_gamma": np.asarray(stage2["cov_gamma"]).tolist(),
            "cov_delta": np.asarray(stage2["cov_delta"]).tolist(),
        },
        "q": idata_ase1.posterior["q"].mean(axis=(0, 1)).data.tolist(),
        "stage2_info": {
            "sigma_u": stage2["sigma_u"],
            "sigma_gamma": stage2["sigma_gamma"],
            "converged": stage2["converged"],
        },
        "pymc_info": {
            "n_chains": len(idata_ase1["sample_stats"]["chain"]),
            "n_samples": len(gene_data.ids),
            "n_ase_informative_samples": len(
                idata_ase1.posterior.coords["fit_samples"]
            ),
            "n_diverging": int(idata_ase1["sample_stats"]["diverging"].sum()),
            "n_diverging_chains": int(
                idata_ase1["sample_stats"]["diverging"].any(axis=1).sum(axis=0)
            ),
            "mean_tree_depth": float(idata_ase1["sample_stats"]["depth"].mean()),
            "median_rhat_eta": eta_summary["r_hat"].median(),
            "min_ess_bulk_eta": eta_summary["ess_bulk"].min(),
            "min_ess_tail_eta": eta_summary["ess_tail"].min(),
        },
    }
