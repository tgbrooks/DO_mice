import polars as pl
import scipy.special
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import scipy.optimize
import formulaic

from util.compat_classes import CompatClassesDf
from util.pymc_helpers import (
    constrained_normal,
    antisymmetric_constraints,
)


def make_ase_model(
    gene_id: str,
    gene_class_counts: CompatClassesDf,
    diplotypes: pl.DataFrame,
    gene_effective_lengths: pl.DataFrame,
    config: dict | None = None,
) -> pm.Model:
    """
    Create a PYMC model object for the ASE model

    gene_effective_lengths should have mouse_id and effective_length (Salmon-style)
        and haplotype columns. Only haplotypes in each mouse's diplotype are needed.
    """

    default_config = {
        "sigma_u_sd": 1.0,
        "sigma_gamma_sd": 1.0,
        "sigma_v_sd": 1.0,
        "dirichlet_scale_sd": 3.0,
        "dirichlet_scale_mu": np.log(1e3),
        "beta_sd": 2.0,
        "use_q_logit": True,
        "q_logit_sd": 2.0,
        "u_centered": True,  # Note: terminology reversed, True = N(0,1)*sigma (non-centered) TODO: fix
        "v_centered": True,
        "use_eta": False,
        "gamma_centered": True,
        "min_genotype_confidence": 0.95,
    }
    if config is not None:
        default_config.update(config)
    config = default_config

    gene_diplotypes = (
        diplotypes.filter(gene_id=gene_id)
        .filter(
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

    HAPLOTYPES = gene_class_counts.haplotypes
    HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}

    hap1 = gene_diplotypes.select(pl.col("hap1").replace_strict(HAP_TO_HAPNUM))[
        "hap1"
    ].to_numpy()
    hap2 = gene_diplotypes.select(pl.col("hap2").replace_strict(HAP_TO_HAPNUM))[
        "hap2"
    ].to_numpy()
    n_classes = int(gene_class_counts.compat.shape[0])
    n_haps = len(HAPLOTYPES)
    n_pairs = (n_haps * (n_haps - 1)) // 2
    pairs = [(a, b) for a in range(n_haps) for b in range(a + 1, n_haps)]
    pair_index = {pr: k for k, pr in enumerate(pairs)}
    hap_het_pairs = [f"pair_{HAPLOTYPES[h1]}{HAPLOTYPES[h2]}" for h1, h2 in pairs]
    class_names = [
        "".join(h for j, h in enumerate(HAPLOTYPES) if gene_class_counts.compat[i, j])
        for i in range(n_classes)
    ]

    n_samples = len(gene_class_counts.ids)
    model = pm.Model(
        coords={
            "haplotypes": HAPLOTYPES,
            "haplotypes2": HAPLOTYPES,
            "classes": class_names,
            "q_coordinates": [f"comp_{i}" for i in range(n_classes)],
            "samples": gene_class_counts.ids,
            "hap_pairs": hap_het_pairs,
        }
    )
    eff_len1 = gene_diplotypes["eff_len1"]
    eff_len2 = gene_diplotypes["eff_len2"]

    # A naive estimation of the per-haplotype class frequencies
    # assuming no ASE, but generally very close.
    q_hat = estimate_class_proportions(gene_class_counts.counts, hap1, hap2, n_haps)
    # Information matrix of softmax-multinomial: proportional to diag(q) - q q^T
    # we use the plugin estimate for q. Used to whiten the coordinates of q.
    M = np.array([np.diag(row) - row[:, None] @ row[None, :] for row in q_hat])
    eig_vals, eig_vecs = np.linalg.eigh(M)
    # Estimate alphas
    _var_fit, alpha = estimate_class_alpha(
        gene_class_counts,
        q_hat,
        hap1,
        hap2,
    )

    # For delta: these are the beta and gamma components of delta
    Bm = np.zeros((len(pairs), n_haps))
    for k, (a, b) in enumerate(pairs):
        Bm[k, a], Bm[k, b] = 1.0, -1.0
    P_B = Bm @ Bm.T / n_haps
    P_G = np.eye(len(pairs)) - P_B

    # per-mouse lookup: homozygotes contribute 0; handle BA ordering with a sign flip
    idx = np.zeros(n_samples, dtype=int)
    sign = np.zeros(n_samples)
    for j, (a, b) in enumerate(zip(hap1, hap2)):
        if a != b:
            idx[j] = pair_index[(min(a, b), max(a, b))]
            sign[j] = 1.0 if a < b else -1.0

    # In order to know whether to use centered or noncentered, we need to know
    # how informative a class is. For high expressed classes, centered works better.
    # for low expressed, non-centered. Each gene may have some of both.
    est_hap_total_counts = np.array(
        [
            (gene_class_counts.counts[(hap1 == i) | (hap2 == i)].sum(axis=0)) / 2
            for i in range(n_haps)
        ]
    )
    est_hap_class_counts = est_hap_total_counts * q_hat

    with model:
        _counts = pm.Data(
            "counts", gene_class_counts.counts, dims=("samples", "classes")
        )
        _hap1 = pm.Data("hap1", hap1, dims="samples")
        _hap2 = pm.Data("hap2", hap2, dims="samples")
        _eff_len1 = pm.Data("eff_len1", eff_len1, dims="samples")
        _eff_len2 = pm.Data("eff_len2", eff_len2, dims="samples")
        _alpha = pm.Data("alpha", alpha, dims=("samples", "classes"))
        _q_hat = pm.Data("q_hat", q_hat, dims=("haplotypes", "classes"))
        _nz = _counts > 0

        # Nuisance variables
        # Rates at which reads from a haplotype are assigned to each class
        # log-scale q values, which get whitened according to M above
        if config["q_mode"] == "eig_vecs":
            q_logit = (
                pm.Normal(
                    "q_logit",
                    dims=("haplotypes", "q_coordinates"),
                )
                * config["q_logit_sd"]
            )
            q = pm.Deterministic(
                "q",
                pm.math.softmax(
                    np.log(_q_hat) + (eig_vecs @ q_logit[:, :, None])[:, :, 0], axis=1
                ),
                dims=("haplotypes", "classes"),
            )
        elif config["q_mode"] == "q_hat":  # just use the plugin value
            q_logit = pm.Deterministic(
                "q_logit", pt.zeros((n_haps, n_classes)), dims=("haplotypes", "classes")
            )
            q = pm.Deterministic(
                "q",
                _q_hat,
                dims=("haplotypes", "classes"),
            )
        elif config["q_mode"] == "centered":
            sigma_q = pm.HalfNormal("sigma_q", sigma=config["sigma_q_sd"])
            q_logit = pm.ZeroSumNormal(
                "q_logit", sigma=sigma_q, dims=("haplotypes", "classes")
            )
            q = pm.Deterministic(
                "q",
                pm.math.softmax(np.log(_q_hat) + q_logit, axis=1),
                dims=("haplotypes", "classes"),
            )
        elif config["q_mode"] == "noncentered":
            sigma_q = pm.HalfNormal("sigma_q", sigma=config["sigma_q_sd"])
            q_logit = pm.ZeroSumNormal(
                "q_logit", sigma=1, dims=("haplotypes", "classes")
            )
            q = pm.Deterministic(
                "q",
                pm.math.softmax(np.log(_q_hat) + sigma_q * q_logit, axis=1),
                dims=("haplotypes", "classes"),
            )
        elif config["q_mode"] == "adaptive_centered":
            # Center high expressers only
            threshold = 50
            center = est_hap_class_counts > threshold
            # Pin the highest class of each haplotype to 0
            pin_idx = np.argmax(est_hap_class_counts, axis=1)
            pin = np.arange(n_classes)[None, :] == pin_idx[:, None]
            # Separate out the tiny classes where the q_hat estimate is unreliable
            tiny = q_hat < 2e-5
            sigma_q = pm.HalfNormal("sigma_q", sigma=config["sigma_q_sd"])
            sigma_q_tiny = pm.HalfNormal("sigma_q_tiny", sigma=config["sigma_q_sd"])
            sigmas = pt.ones(center.shape)
            sigmas = pt.set_subtensor(sigmas[center & (~tiny)], sigma_q)
            sigmas = pt.set_subtensor(sigmas[center & tiny], sigma_q_tiny)
            q_logit = pm.Normal("q_logit", sigma=sigmas, dims=("haplotypes", "classes"))
            noncenter_sigmas = pt.ones(center.shape)
            noncenter_sigmas = pt.set_subtensor(sigmas[~center & (~tiny)], sigma_q)
            noncenter_sigmas = pt.set_subtensor(sigmas[~center & (tiny)], sigma_q_tiny)
            q_logit_final = (noncenter_sigmas * q_logit) * (~pin)
            q = pm.Deterministic(
                "q",
                pm.math.softmax(np.log(_q_hat) + q_logit_final, axis=1),
                dims=("haplotypes", "classes"),
            )
        else:
            raise ValueError("Unrecognized 'q_mode'")

        # # Haplotype effects
        # beta = pm.ZeroSumNormal("beta", sigma=config["beta_sd"], dims="haplotypes")

        # # Diplotype effects
        # # Deviations from additive haplotype effects, constrained to be orthogonal
        # # to the additive effect
        # sigma_gamma = pm.HalfNormal("sigma_gamma") * config["sigma_gamma_sd"]
        # # Orthogonal to the beta's
        # orthog_to_beta = np.zeros((n_haps, n_haps, n_haps))
        # for i in range(n_haps):
        #     orthog_to_beta[i, i, :] += 1
        #     orthog_to_beta[i, :, i] -= 1
        # gamma_null = np.concat((orthog_to_beta, antisymmetric_constraints(n_haps)))
        # if config["gamma_centered"]:
        #     gamma = constrained_normal(
        #         "gamma",
        #         orthog_to=gamma_null,
        #         sigma=1.0,
        #         dims=("haplotypes", "haplotypes2"),
        #     )
        #     _gamma = sigma_gamma * gamma
        # else:
        #     gamma = constrained_normal(
        #         "gamma",
        #         orthog_to=gamma_null,
        #         sigma=sigma_gamma,
        #         dims=("haplotypes", "haplotypes2"),
        #     )
        #     _gamma = gamma

        ## Delta: union of gamma and beta, one variable per hap pair (hets only)
        sigma_gamma = pm.HalfNormal("sigma_gamma") * config["sigma_gamma_sd"]
        delta_cov = n_haps * config["beta_sd"] ** 2 * P_B + sigma_gamma**2 * P_G
        delta = pm.MvNormal(
            "delta", mu=np.zeros(len(pairs)), cov=delta_cov, dims="hap_pairs"
        )
        beta = pm.Deterministic("beta", Bm.T @ delta / n_haps, dims="haplotypes")
        gamma_raw = pm.Deterministic("gamma_raw", P_G @ delta, dims="hap_pairs")

        sigma_u = pm.HalfNormal("sigma_u") * config["sigma_u_sd"]
        lin = sign * delta[idx] + pm.math.log(_eff_len1) - pm.math.log(_eff_len2)
        eta = pm.Normal("eta", mu=lin, sigma=sigma_u, dims="samples")
        u = pm.Deterministic("u", eta - lin, dims="samples")

        # lin = (
        #    beta[_hap1]
        #    - beta[_hap2]
        #    + pm.math.log(_eff_len1)
        #    - pm.math.log(_eff_len2)
        #    + _gamma[_hap1, _hap2]
        # )
        # # Random per-sample effects
        # sigma_u = pm.HalfNormal("sigma_u") * config["sigma_u_sd"]
        # if config["use_eta"]:
        #    eta = pm.Normal("eta", mu=lin, sigma=sigma_u, dims="samples")
        #    u = pm.Deterministic("u", eta - lin, dims="samples")
        # else:
        #    if config["u_centered"]:
        #        u_raw = pm.Normal("u_raw", 0, 1, dims="samples")
        #        u = sigma_u * u_raw
        #    else:
        #        u = pm.Normal("u", 0, sigma=sigma_u, dims="samples")
        #    eta = lin + u

        p = pm.Deterministic(
            "p",
            pm.math.sigmoid(eta),
            dims=("samples"),
        )

        # # Try adding log-normal noise to each class
        # # TODO: log or logit here?
        # log_mu = pm.math.log(p[:, None] * q[_hap1] + (1 - p)[:, None] * q[_hap2])
        # sigma_v = pm.HalfNormal("sigma_v", sigma=config["sigma_v_sd"])
        # # v = pm.Normal("v", mu=class_props, sigma=sigma_v, dims=("samples", "classes"))
        # # Split whether we do centered or non-centered v by counts for efficiency
        # counts_np = gene_class_counts.counts
        # hi = counts_np >= 50  # roughly: centered when count * sigma_v**2 >~ 1
        # i_hi, i_lo = np.nonzero(hi), np.nonzero(~hi)
        # v_hi = pm.Normal("v_hi", mu=log_mu[i_hi], sigma=sigma_v, shape=int(hi.sum()))
        # z_lo = pm.Normal("z_lo", 0, 1, shape=int((~hi).sum()))
        # v = pt.zeros_like(log_mu)
        # v = pt.set_subtensor(v[i_hi], v_hi)
        # v = pt.set_subtensor(v[i_lo], log_mu[i_lo] + sigma_v * z_lo)

        # Original
        # class_props = p[:, None] * q[_hap1] + (1 - p)[:, None] * q[_hap2]
        # pm.Potential("ll", pt.sum(_counts[_nz] * pt.log(class_props[_nz])))

        # Dirichlet multinomial version
        # Log-likelihood - used instaed of pm.Multinomial since its a bit faster
        # We drop the constant terms that don't depend upon class_props
        # pm.Potential("ll", pt.sum(_counts[_nz] * pt.log(class_props[_nz])))
        # class_props = p[:, None] * q[_hap1] + (1 - p)[:, None] * q[_hap2]
        # dirichlet_scale = pm.Normal(
        #    "dirichlet_scale",
        #    mu=config["dirichlet_scale_mu"],
        #    sigma=config["dirichlet_scale_sd"],
        # )
        # pm.DirichletMultinomial(
        #    "dm_counts",
        #    n=pt.sum(_counts, axis=1),
        #    a=class_props * np.exp(dirichlet_scale),
        #    dims=("samples", "classes"),
        #    observed=_counts,
        # )

        # Normal
        class_props = p[:, None] * q[_hap1] + (1 - p)[:, None] * q[_hap2]
        totals = gene_class_counts.counts.sum(axis=1, keepdims=True)
        # sigma_final_coeffs = pm.Normal("sigma_final_coeffs", 0.0, 2.0, shape=B.shape[1])
        # pm.Normal(
        #     "y",
        #     mu=np.log(totals * class_props + 1),
        #     sigma=pt.exp(B @ sigma_final_coeffs).reshape((n_samples, n_classes)),
        #     observed=np.log(_counts + 1),
        #     dims=("samples", "classes"),
        # )

        # Negative-binomial option
        # kappa = pm.LogNormal("kappa", 0.0, 1.0)
        # alpha_final_coeffs = pm.Normal("alpha_final_coeffs", 0.0, 2.0, shape=B.shape[1])
        # pm.NegativeBinomial(
        #    "y",
        #    mu=totals * class_props,
        #    alpha=alpha,
        #    # alpha=np.exp(B @ alpha_final_coeffs).reshape((n_samples, n_classes)),
        #    observed=_counts,
        #    dims=("samples", "classes"),
        # )

        # Negative-binomial as a potential - samples much faster for unclear reasons
        mu = totals * class_props
        nb_const = (
            pt.gammaln(_alpha + gene_class_counts.counts)
            - pt.gammaln(gene_class_counts.counts + 1)
            - pt.gammaln(_alpha)
        )
        pm.Potential(
            "ll",
            pt.sum(
                nb_const
                # Equivalent to: log[ (alpha / (alpha + mu))^alpha (mu / (alpha + mu))^counts ]
                # but slightly more numerically stable for very large alpha (i.e., near Poisson)
                - _alpha * pt.log1p(mu / _alpha)
                + _counts * (pt.log(mu) - pt.log(_alpha + mu))
            ),
        )

    return model


def estimate_class_proportions(class_counts, hap1, hap2, n_haps):
    """Simple estimate of q_gi (the proportion of reads from haplotype g going to class i) assuming NO ASE

    So p_j = 1/2 for all samples j. This reduces to a linear equation.

    c_ji proportional to (q_{g1,i} + q_{g2,i})/2
    """
    class_props = class_counts / (class_counts.sum(axis=1)[:, None] + 1e-9)
    n_samples, n_classes = class_counts.shape
    X = np.zeros((n_samples, n_haps))
    for g in range(n_haps):
        X[hap1 == g, g] += 1 / 2
        X[hap2 == g, g] += 1 / 2
    q_hat = np.array(
        [scipy.optimize.nnls(X, class_props[:, i])[0] for i in range(n_classes)]
    ).T
    # Force not too tiny to be conservative
    q_hat[q_hat < 1e-5] = 1e-5
    q_hat /= q_hat.sum(axis=1)[:, None]
    return q_hat  # n_haps x n_classes


def estimate_class_alpha(gene_class_counts, q_hat, hap1, hap2):
    """
    Compute a trend line of variance over class proportion
    Use that to compute alpha (1/dispersion) for each class's counts

    q_hat from estimate_class_proportions()
    """
    temp = []
    haps = pl.DataFrame({"mouse_id": gene_class_counts.ids, "hap1": hap1, "hap2": hap2})
    # NOTE: this works best if diplotypes are always sorted (AB not BA), which they are in
    # our dataset but may not be in general (e.g., simulated). Estimate is just noisier otherwise.
    for (h1, h2), dat in haps.group_by(["hap1", "hap2"]):
        these_mice = dat["mouse_id"]
        these_mice_ids = np.where(np.isin(gene_class_counts.ids, these_mice))
        if len(these_mice) < 4:
            # Variance too high in this diplotype, ignore
            continue
        these_counts = gene_class_counts.counts[these_mice_ids]
        tots = np.sum(these_counts, axis=1)
        q_hat_close = np.abs(np.log(q_hat[h1] / q_hat[h2])) < 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            props = these_counts / np.sum(these_counts, axis=1, keepdims=True)
            mprops = np.mean(props, axis=0)
            true_var = props.var(axis=0, ddof=1)
        temp.append(
            pl.DataFrame(
                {
                    "hap1": h1,
                    "hap2": h2,
                    "var": true_var,
                    "mean_T": np.mean(tots),
                    "mean_inv_T": np.mean(1 / tots),
                    "q_hat_close": q_hat_close,
                    "log_prop": np.log(mprops),
                    "n_samples": len(these_mice),
                    "class_num": np.arange(mprops.shape[0]),
                }
            )
        )
    df = pl.concat(temp)
    _df = df.filter("q_hat_close", pl.col("log_prop").is_finite())
    # Fit trend curve
    X = formulaic.Formula(
        "~ bs(log_prop, df=4, extrapolation='clip')"
    ).get_model_matrix(_df)
    y = np.log(_df["var"])
    var_coeffs, _, _, _ = np.linalg.lstsq(X.to_numpy(), y)
    # Now get sample-level trend-curve values
    q_hat_mean = (q_hat[hap1] + q_hat[hap2]) / 2
    X2 = X.model_spec.get_model_matrix(
        pl.DataFrame({"log_prop": np.log(q_hat_mean.ravel())})
    )
    var_fit = np.exp(X2 @ var_coeffs).reshape(q_hat_mean.shape)
    mean_T_inv = np.mean(1 / gene_class_counts.counts.sum(axis=1, keepdims=True))
    alpha = q_hat_mean**2 / (var_fit - q_hat_mean * mean_T_inv)
    alpha[alpha < 0] = 1e3  # Poisson-like fit -> high alpha -> low dispersion
    return var_fit, alpha


def _summary(idata, **kwargs):
    return pl.DataFrame(az.summary(idata, **kwargs).reset_index(names="variable"))


def summarize_ase_model(idata, gene_class_counts):
    """Summarizes the posterior distribution from data sampled from an ASE model"""

    # Prioritize beta values since they're the values we actually care about
    beta_summary = _summary(idata, var_names=["beta"])
    # Secondarily important values
    # We choose gamma_raw over gamma since gamma contains meaningless values like gamma[A,A]
    core_summary = _summary(idata, var_names=["sigma_gamma", "sigma_u", "gamma_raw"])
    # The rest are further less important.
    q_summary = _summary(idata, var_names=["q"])
    # Classes that are extremely rare aren't very interesting and tend to be poorly behaved
    q_summary_important = q_summary.filter(pl.col("mean") > 5e-4)
    if "u_raw" in idata["posterior"].keys():
        u_summary = _summary(idata, var_names=["u_raw"])  # centered version
    elif "u" in idata["posterior"].keys():
        u_summary = _summary(idata, var_names=["u"])  # non-centered version
    else:
        u_summary = _summary(idata, var_names=["eta"])  # non-centered version

    read_classes = summarize_read_classes(idata, gene_class_counts)
    bias_rates = summarize_bias_rates(idata, gene_class_counts)

    return {
        "beta": beta_summary.rows_by_key("variable", unique=True, named=True),
        "core": core_summary.rows_by_key("variable", unique=True, named=True),
        "read_classes": read_classes.to_dicts(),
        "max_leak": read_classes["leak_to"].max(),
        "bias_rates": bias_rates.to_dicts(),
        "max_bias_rate": bias_rates["allele_unique_bias"].max(),
        "diagnostic": {
            "max_rhat_beta": beta_summary["r_hat"].max(),
            "max_rhat_core": core_summary["r_hat"].max(),
            "max_rhat_rest": max(
                q_summary_important["r_hat"].max()
                or 1.0,  # if no classes, all are well-behaved
                u_summary["r_hat"].max(),
            ),
            "min_ess_bulk_beta": beta_summary["ess_bulk"].min(),
            "min_ess_bulk_core": core_summary["ess_bulk"].min(),
            "min_ess_bulk_rest": min(
                q_summary_important["ess_bulk"].min(),
                u_summary["ess_bulk"].min(),
            ),
            "min_ess_tail_beta": beta_summary["ess_tail"].min(),
            "min_ess_tail_core": core_summary["ess_tail"].min(),
            "min_ess_tail_rest": min(
                q_summary_important["ess_tail"].min(),
                u_summary["ess_tail"].min(),
            ),
        },
        "info": {
            "n_chains": len(idata["sample_stats"]["chain"]),
            "n_samples": idata["constant_data"]["counts"].shape[0],
            "n_classes": idata["constant_data"]["counts"].shape[1],
            "n_diverging": int(idata["sample_stats"]["diverging"].sum()),
            "n_diverging_chains": int(
                idata["sample_stats"]["diverging"].any(axis=1).sum(axis=0)
            ),
            "mean_tree_depth": float(idata["sample_stats"]["depth"].mean()),
            "maxdepth_reached_fraction": float(
                idata["sample_stats"]["maxdepth_reached"].mean()
            ),
            "sampling_time": idata["posterior"].attrs["sampling_time"],
        },
    }


def summarize_read_classes(idata, gene_class_counts: CompatClassesDf) -> pl.DataFrame:
    """
    Summarize the mapping classes generated by each haplotype

    Columns:
        source_hap: haplotype that is the source of these reads
        out_class: fraction from source_hap aligning to out_hap
        leak_to: fraction from source_hap aligning to out_hap that do not align to source_hap
        out_hap: haplotype to which these reads align. If multiple, the read is counted
            for each out_hap and therefore we do expect it to sum to 1.
    """
    class_summaries = (idata["posterior"]["q"].values @ gene_class_counts.compat).mean(
        axis=(0, 1)
    )
    HAPLOTYPES = gene_class_counts.haplotypes
    leak_outs = np.array(
        [
            (
                idata["posterior"]["q"].values[:, :, i, :]
                @ ((~gene_class_counts.compat[:, i, None]) & gene_class_counts.compat)
            ).mean(axis=(0, 1))
            for i in range(len(HAPLOTYPES))
        ]
    )
    class_summaries = pl.concat(
        [
            pl.DataFrame(
                {
                    "source_hap": hap,
                    "out_class": class_summaries[i],
                    "leak_to": leak_outs[i],
                    "out_hap": HAPLOTYPES,
                }
            )
            for i, hap in enumerate(HAPLOTYPES)
        ]
    )
    return class_summaries


def summarize_bias_rates(idata_ase, gene_class_counts: CompatClassesDf) -> pl.DataFrame:
    """
    Summarize the bias rates per diplotype in the fraction of allele-unique reads generated
    by each haplotype.

    Columns:
        source_hap: haplotype that is the source of these reads
        other_hap: second haplotype in the diplotype
        frac_unique: fraction of reads from source_hap that do not align to other_hap
        reverse_frac_unique: fraction of reads from other_hap that do not align to source_hap
        allele_unique_bias: bias ratio of frac_unique / reverse_frac_unique
            1 means no bias, higher values means source_hap is over represented in unique reads.
    """
    HAPLOTYPES = gene_class_counts.haplotypes
    q = idata_ase["posterior"]["q"].values
    temp = []
    for i, h1 in enumerate(HAPLOTYPES):
        for j, h2 in enumerate(HAPLOTYPES):
            h1_unique_classes = gene_class_counts.compat[:, i] & (
                ~gene_class_counts.compat[:, j]
            )
            h1_u = q[..., i, h1_unique_classes].sum(axis=-1).mean(axis=(0, 1))

            h2_unique_classes = gene_class_counts.compat[:, j] & (
                ~gene_class_counts.compat[:, i]
            )
            h2_u = q[..., j, h2_unique_classes].sum(axis=-1).mean(axis=(0, 1))
            temp.append(
                {
                    "source_hap": h1,
                    "other_hap": h2,
                    "frac_unique": h1_u,
                    "reverse_frac_unique": h2_u,
                    "allele_unique_bias": (h1_u + 1e-9) / (h2_u + 1e-9),
                }
            )
    return pl.DataFrame(temp).filter(pl.col("source_hap") != pl.col("other_hap"))
