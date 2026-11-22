"""
Joint allele-specific expression (ASE) model, one pass.

Same read-class layer (q) and count likelihood as stage 1 of the two-stage model, but
each het mouse's allelic log-ratio is tied to the haplotype effects directly:

    eta_j  = sign_j * delta[pair_j] + u_j    (expression log-ratio, hap1 vs hap2)
    p_j    = sigmoid(eta_j + offset_j)       (hap1's share of the reads)
    delta  = Bm beta + gamma                 (one delta per unordered haplotype pair)
    beta   ~ ZeroSumNormal(sigma_beta)       (the effects we care about)
    gamma  = sigma_gamma * G z_gamma,  z_gamma ~ N(0, I)   (non-additive pair effects)
    u_j    = sigma_u * z_j,            z_j     ~ N(0, 1)   (per-mouse deviation)

eta is the biological allelic imbalance, at the level of transcripts. offset_j is the
log ratio of the two haplotypes' effective lengths (log_eff_ratio), which turns it into
the imbalance in reads; it is stored in the samples' constant_data.

G is an orthonormal basis of the pair-effect directions orthogonal to the additive
ones (range of P_G), so gamma never changes beta. gamma and u are non-centered: each
pair (~7 mice) and each mouse informs its own deviation only weakly relative to the
scales, which is the regime where non-centering samples well.

The beta posterior (mean and covariance of the draws) is used directly; there is no
handoff between stages. q, gamma, u and eta are for QC only.
"""

import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt

from ase_buffering.ase_model_2steps import GeneData, _make_q, _nb_potential


DEFAULTS = dict(
    sigma_q=1.0,  # prior sd of q logits around q_hat
    sigma_beta=2.0,
    sigma_gamma_prior_sd=0.5,  # HalfNormal scale for sigma_gamma
    sigma_u_prior_sd=0.5,  # HalfNormal scale for sigma_u
    sigma_log_total_reads=1.0,
    min_n_het=50,  # fail genes with fewer heterozygous mice than this
    target_accept=0.9,  # 0.8 gave noticeably worse R-hat / ESS on simulated data
)


def make_joint_model(d: GeneData, config: dict) -> pm.Model:
    config = {**DEFAULTS, **config}
    i_het = np.nonzero(d.sign != 0)[0]
    if len(i_het) < config["min_n_het"]:
        raise ValueError(f"Only {len(i_het)} heterozygous mice")

    # Orthonormal basis of the non-additive pair directions (P_G is a projection)
    w, V = np.linalg.eigh(d.P_G)
    G = V[:, w > 0.5]  # n_pairs x (n_pairs - n_haps + 1)

    coords = {
        "samples": d.ids,
        "het_samples": [d.ids[i] for i in i_het],
        "haplotypes": d.haplotypes,
        "classes": d.class_names,
        "pairs": d.pair_names,
        "gamma_basis": np.arange(G.shape[1]),
    }
    with pm.Model(coords=coords) as model:
        q = _make_q(d, config)

        beta = pm.ZeroSumNormal("beta", sigma=config["sigma_beta"], dims="haplotypes")
        sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=config["sigma_gamma_prior_sd"])
        z_gamma = pm.Normal("z_gamma", dims="gamma_basis")
        gamma = pm.Deterministic(
            "gamma", sigma_gamma * pt.dot(G, z_gamma), dims="pairs"
        )
        delta = pm.Deterministic("delta", pt.dot(d.Bm, beta) + gamma, dims="pairs")

        sigma_u = pm.HalfNormal("sigma_u", sigma=config["sigma_u_prior_sd"])
        z_u = pm.Normal("z_u", dims="het_samples")
        u = pm.Deterministic("u", sigma_u * z_u, dims="het_samples")

        # Biological allelic log-ratio (hap1 vs hap2), without effective-length effects
        eta_het = pm.Deterministic(
            "eta", d.sign[i_het] * delta[d.idx[i_het]] + u, dims="het_samples"
        )
        log_eff_ratio = pm.Data(
            "log_eff_ratio", d.log_eff_ratio[i_het].astype(float), dims="het_samples"
        )
        # Allelic log-ratio of the reads. Homozygotes: p does not enter the likelihood
        # (q[hap1] == q[hap2]), so their value is arbitrary.
        eta_reads = pt.as_tensor(d.log_eff_ratio.astype(float))
        eta_reads = pt.set_subtensor(eta_reads[i_het], eta_het + log_eff_ratio)
        p = pm.math.sigmoid(eta_reads)

        class_props = p[:, None] * q[d.hap1] + (1 - p)[:, None] * q[d.hap2]
        log_total_reads = pm.Normal(
            "log_total_reads",
            mu=np.log(d.counts.sum(axis=1)),
            sigma=config["sigma_log_total_reads"],
            dims="samples",
        )
        _nb_potential(d.counts, d.alpha, pt.exp(log_total_reads[:, None]) * class_props)
    return model


def _convergence(post, var, mask=None):
    """Worst R-hat / ESS over the entries of `var` (only where `mask`, if given)."""
    # Select the variable rather than pass var_names: arviz 1.3 fails to find scalar
    # variables by var_names
    sub = post[[var]]
    ess_bulk = np.atleast_1d(az.ess(sub, method="bulk")[var].values)
    ess_tail = np.atleast_1d(az.ess(sub, method="tail")[var].values)
    rhat = np.atleast_1d(az.rhat(sub)[var].values)
    if mask is not None:
        ess_bulk, ess_tail, rhat = ess_bulk[mask], ess_tail[mask], rhat[mask]
    return dict(
        max_rhat=float(np.max(rhat)),
        min_ess_bulk=float(np.min(ess_bulk)),
        min_ess_tail=float(np.min(ess_tail)),
    )


# q entries with less posterior mass than this are left out of the q diagnostics: their
# logits (q_hat floored at 1e-5) are barely touched by the data and wander harmlessly.
MIN_Q_FOR_DIAGNOSTICS = 5e-4


def summarize_joint_model(idata) -> dict:
    """beta and its posterior covariance (the results), plus QC and convergence.

    Uses only the sampled InferenceData / DataTree, so it can run from a saved .nc.
    """
    post = idata.posterior
    post = post.to_dataset() if hasattr(post, "to_dataset") else post
    beta_draws = post["beta"].stack(s=("chain", "draw")).transpose("haplotypes", "s")
    stats = idata.sample_stats
    stats = stats.to_dataset() if hasattr(stats, "to_dataset") else stats

    def mean(var):
        return post[var].mean(("chain", "draw")).values.tolist()

    def sd(var):
        return post[var].std(("chain", "draw")).values.tolist()

    const = idata.constant_data
    const = const.to_dataset() if hasattr(const, "to_dataset") else const

    return dict(
        beta=beta_draws.mean("s").values.tolist(),
        cov_beta=np.cov(beta_draws.values).tolist(),
        sigma_u=float(post["sigma_u"].mean()),
        sigma_gamma=float(post["sigma_gamma"].mean()),
        sigma_u_sd=float(post["sigma_u"].std()),
        sigma_gamma_sd=float(post["sigma_gamma"].std()),
        gamma=mean("gamma"),
        # Lists rather than a dict so every gene has the same fields (parquet-friendly)
        u=dict(mouse_id=post["u"]["het_samples"].values.tolist(), mean=mean("u")),
<<<<<<< Updated upstream
        # Biological allelic log-ratio (hap1 vs hap2 of the sorted diplotype) per het
        # mouse; eta + log_eff_ratio is the log-ratio in reads
        eta=dict(
            mouse_id=post["eta"]["het_samples"].values.tolist(),
            mean=mean("eta"),
            sd=sd("eta"),
            log_eff_ratio=const["log_eff_ratio"].values.tolist(),
        ),
=======
        eta=dict(mouse_id=post["eta"]["het_samples"].values.tolist(), mean=mean("eta")),
>>>>>>> Stashed changes
        q=mean("q"),
        # Per-chain beta means: chains settling on different solutions show up here
        beta_chain_means=post["beta"].mean("draw").values.tolist(),
        diagnostics=dict(
            n_samples=int(post.sizes["samples"]),
            n_het=int(post.sizes["het_samples"]),
            n_chains=int(post.sizes["chain"]),
            n_draws=int(post.sizes["draw"]),
            n_divergences=int(stats["diverging"].sum()),
            mean_tree_depth=float(stats["tree_depth"].mean())
            if "tree_depth" in stats
            else float(stats["depth"].mean()),
            beta=_convergence(post, "beta"),
            sigma_u=_convergence(post, "sigma_u"),
            sigma_gamma=_convergence(post, "sigma_gamma"),
            u=_convergence(post, "u"),
            eta=_convergence(post, "eta"),
            q=_convergence(
                post,
                "q",
                mask=post["q"].mean(("chain", "draw")).values > MIN_Q_FOR_DIAGNOSTICS,
            ),
        ),
    )
