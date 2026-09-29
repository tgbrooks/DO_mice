import polars as pl
import numpy as np
import pymc as pm
import pytensor.tensor as pt

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
        q_logit = pm.Normal("q_logit", sigma=2, dims=("haplotypes", "classes"))
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
    X = np.zeros((n_samples, n_haps))
    for g in range(n_haps):
        X[hap1 == g, g] += 1 / 2
        X[hap2 == g, g] += 1 / 2
    q_hat, _, _, _ = np.linalg.lstsq(X, class_props[:, :], rcond=None)
    # Force non-negative and not too tiny to be conservative
    q_hat[q_hat < 1e-5] = 1e-5
    q_hat /= q_hat.sum(axis=1)[:, None]
    return q_hat


def summarize_ase_model(idata):
    """Summarizes the posterior distribution from data sampled from an ASE model"""
