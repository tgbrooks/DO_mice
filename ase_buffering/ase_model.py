import polars as pl
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from util.compat_classes import CompatClassesDf


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
    model = pm.Model(
        coords={
            "haplotypes": HAPLOTYPES,
            "classes": [f"class_{i}" for i in range(n_classes)],
            "samples": gene_class_counts.ids,
        }
    )
    # mask out classes which have no expression from anything with that haplotype
    mask = np.array(
        [
            gene_class_counts.counts[hap1 == i].any(axis=0).astype(int)
            | gene_class_counts.counts[hap2 == i].any(axis=0).astype(int)
            for i in range(len(HAPLOTYPES))
        ]
    )

    def vector_to_antisymmetric(v, n):
        """Rearrange into a zero-diagonal, symmetric matrix"""
        matrix = pt.zeros((n, n), dtype=v.dtype)
        triu_indices = np.triu_indices(n, k=1)
        matrix = pt.set_subtensor(matrix[triu_indices], v)
        return matrix - matrix.T

    with model:
        _counts = pm.Data(
            "counts", gene_class_counts.counts, dims=("samples", "classes")
        )
        _hap1 = pm.Data("hap1", hap1, dims="samples")
        _hap2 = pm.Data("hap2", hap2, dims="samples")
        _mask = pm.Data("mask", mask, dims=("haplotypes", "classes"))
        _nz = _counts > 0

        # Nuisance variables
        # Rates at which reads from a haplotype are assigned to each class
        q_raw = pm.Normal("q_raw", sigma=3, dims=("haplotypes", "classes"))

        def masked_softmax(q, mask):
            exp = pm.math.exp(q) * mask
            norm = exp.sum(axis=1)[:, None]
            return exp / norm

        q = pm.Deterministic(
            "q", masked_softmax(q_raw, _mask), dims=("haplotypes", "classes")
        )

        # Haplotype effects
        beta = pm.ZeroSumNormal("beta", sigma=2.0, dims="haplotypes")

        # Diplotype effects
        sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=0.05)
        # sigma_gamma = 0
        gamma = pm.Normal("gamma", sigma=1.0, shape=(n_haps * (n_haps - 1) / 2,))
        gamma_full = vector_to_antisymmetric(sigma_gamma * gamma, n_haps)

        # Random per-sample effects
        sigma_u = pm.HalfNormal("sigma_u", 0.03)
        u_raw = pm.Normal("u_raw", 0, 1, dims="samples")
        u = sigma_u * u_raw

        p = pm.Deterministic(
            "p",
            pm.math.sigmoid(beta[_hap1] - beta[_hap2] + gamma_full[_hap1, _hap2] + u),
            dims=("samples"),
        )
        class_props = p[:, None] * q[_hap1] + (1 - p)[:, None] * q[_hap2]

        # Log-likelihood - used instaed of pm.Multinomial since its a bit faster
        # We drop the constant terms that don't depend upon class_props
        pm.Potential("ll", pt.sum(_counts[_nz] * pt.log(class_props[_nz])))

    return model


def summarize_ase_model(idata):
    """Summarizes the posterior distribution from data sampled from an ASE model"""
