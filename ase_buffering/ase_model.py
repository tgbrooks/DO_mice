import polars as pl
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import scipy.optimize

from util.compat_classes import CompatClassesDf
from util.pymc_helpers import (
    constrained_normal,
    antisymmetric_constraints,
)


def make_ase_model(
    gene_id: str,
    gene_class_counts: CompatClassesDf,
    diplotypes: pl.DataFrame,
) -> pm.Model:
    """Create a PYMC model object for the ASE model"""

    gene_diplotypes = (
        diplotypes.filter(gene_id=gene_id)
        .select("diplotype", "mouse_id")
        .join(
            pl.DataFrame({"mouse_id": gene_class_counts.ids}),
            "mouse_id",
            maintain_order="right",
        )
    )
    HAPLOTYPES = gene_class_counts.haplotypes
    HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}

    hap1 = gene_diplotypes.select(
        pl.col("diplotype").str.slice(0, 1).replace_strict(HAP_TO_HAPNUM)
    )["diplotype"].to_numpy()
    hap2 = gene_diplotypes.select(
        pl.col("diplotype").str.slice(1, 1).replace_strict(HAP_TO_HAPNUM)
    )["diplotype"].to_numpy()
    n_classes = int(gene_class_counts.compat.shape[0])
    n_haps = len(HAPLOTYPES)
    n_pairs = (n_haps * (n_haps - 1)) // 2
    model = pm.Model(
        coords={
            "haplotypes": HAPLOTYPES,
            "haplotypes2": HAPLOTYPES,
            "classes": [f"class_{i}" for i in range(n_classes)],
            "q_coordinates": [f"comp_{i}" for i in range(n_classes)],
            "samples": gene_class_counts.ids,
            "hap_pairs": [f"pair_{i}" for i in range(n_pairs)],
        }
    )

    # A naive estimation of the per-haplotype class frequencies
    # assuming no ASE, but generally very close.
    q_hat = estimate_class_proportions(gene_class_counts.counts, hap1, hap2, n_haps)
    # Information matrix of softmax-multinomial: proportional to diag(q) - q q^T
    # we use the plugin estimate for q. Used to whiten the coordinates of q.
    M = np.array([np.diag(row) - row[:, None] @ row[None, :] for row in q_hat])
    eig_vals, eig_vecs = np.linalg.eigh(M)

    with model:
        _counts = pm.Data(
            "counts", gene_class_counts.counts, dims=("samples", "classes")
        )
        _hap1 = pm.Data("hap1", hap1, dims="samples")
        _hap2 = pm.Data("hap2", hap2, dims="samples")
        _nz = _counts > 0

        # Nuisance variables
        # Rates at which reads from a haplotype are assigned to each class
        # log-scale q values, which get whitened according to M above
        q_logit = pm.Normal("q_logit", sigma=2, dims=("haplotypes", "q_coordinates"))
        q = pm.Deterministic(
            "q",
            pm.math.softmax(
                np.log(q_hat) + (eig_vecs @ q_logit[:, :, None])[:, :, 0], axis=1
            ),
            dims=("haplotypes", "classes"),
        )

        # Haplotype effects
        beta = pm.ZeroSumNormal("beta", sigma=2.0, dims="haplotypes")

        # Diplotype effects
        # Deviations from additive haplotype effects, constrained to be orthogonal
        # to the additive effect
        sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=1.0)
        # Orthogonal to the beta's
        orthog_to_beta = np.zeros((n_haps, n_haps, n_haps))
        for i in range(n_haps):
            orthog_to_beta[i, i, :] += 1
            orthog_to_beta[i, :, i] -= 1
        gamma_null = np.concat((orthog_to_beta, antisymmetric_constraints(n_haps)))
        gamma = constrained_normal(
            "gamma", orthog_to=gamma_null, sigma=1.0, dims=("haplotypes", "haplotypes2")
        )

        # Random per-sample effects
        sigma_u = pm.HalfNormal("sigma_u", 1.0)
        u_raw = pm.Normal("u_raw", 0, 1, dims="samples")
        u = sigma_u * u_raw

        p = pm.Deterministic(
            "p",
            pm.math.sigmoid(
                beta[_hap1] - beta[_hap2] + sigma_gamma * gamma[_hap1, _hap2] + u
            ),
            dims=("samples"),
        )
        class_props = p[:, None] * q[_hap1] + (1 - p)[:, None] * q[_hap2]

        # Log-likelihood - used instaed of pm.Multinomial since its a bit faster
        # We drop the constant terms that don't depend upon class_props
        pm.Potential("ll", pt.sum(_counts[_nz] * pt.log(class_props[_nz])))

    return model


def estimate_class_proportions(class_counts, hap1, hap2, n_haps):
    """Simple estimate of q_gi (the proportion of reads from haplotype g going to class i) assuming NO ASE

    So p_j = 1/2 for all samples j. This reduces to a linear equation.

    c_ji proportional to (q_{g1,i} + q_{g2,i})/2
    """
    class_props = class_counts / class_counts.sum(axis=1)[:, None]
    n_samples, n_classes = class_counts.shape
    # inverse variance-weights
    weight = np.sqrt(class_counts.sum(axis=1))
    X = np.zeros((n_samples, n_haps))
    for g in range(n_haps):
        X[hap1 == g, g] += 1 / 2
        X[hap2 == g, g] += 1 / 2
    q_hat = np.array(
        [
            scipy.optimize.nnls(weight[:, None] * X, weight * class_props[:, i])[0]
            for i in range(n_classes)
        ]
    ).T
    # Force not too tiny to be conservative
    q_hat[q_hat < 1e-5] = 1e-5
    q_hat /= q_hat.sum(axis=1)[:, None]
    return q_hat  # n_haps x n_classes


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
    u_summary = _summary(idata, var_names=["u_raw"])

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
                q_summary_important["r_hat"].max(),
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
            "n_samples": idata["constant_data"]["counts"].shape[0],
            "n_classes": idata["constant_data"]["counts"].shape[1],
            "n_diverging": int(idata["sample_stats"]["diverging"].sum()),
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
                    "allele_unique_bias": h1_u / h2_u,
                }
            )
    return pl.DataFrame(temp).filter(pl.col("source_hap") != pl.col("other_hap"))
