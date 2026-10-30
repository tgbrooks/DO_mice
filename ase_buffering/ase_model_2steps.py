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
    cov_shrink=0.1,  # shrink stage-1 posterior correlations toward 0 (full mode)
    max_abs_skew=0.5,  # flag mice whose stage-1 posterior is clearly non-Gaussian
    # --- stage 2 ---
    gamma_mode="mvn",  # or "zero"
    sigma_beta=2.0,
    sigma_gamma_sd=1.0,
    sigma_u_sd=0.3,
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
            pl.col("total_counts") > 0 if config["drop_all_zeros"] else True,
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


def fit_stage2_eb(d, lik, sigma_beta=0.5, bound=5.0):
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
        "beta": stage2["beta"].tolist(),
        "gamma": stage2["gamma"].tolist(),
        "delta": stage2["delta"].tolist(),
        "eta": eta_summary.rows_by_key("variable", unique=True, named=True),
        "cov": {
            "cov_beta": stage2["cov_beta"].tolist(),
            "cov_gamma": stage2["cov_gamma"].tolist(),
            "cov_delta": stage2["cov_delta"].tolist(),
        },
        "q": idata_ase1.posterior["q"].mean(axis=(0, 1)).tolist(),
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
