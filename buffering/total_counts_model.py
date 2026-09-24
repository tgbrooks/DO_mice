import polars as pl
import numpy as np
import pymc as pm


from util.compat_classes import CompatClassesDf, get_gene_totals

SEX_TO_NUM = {"F": 0, "M": 1}


def make_total_model(
    gene_id: str,
    gene_class_counts: CompatClassesDf,
    diplotypes: pl.DataFrame,
    pheno: pl.DataFrame,
) -> pm.Model:
    """Create a model of the total counts

    diplotypes should have columns mouse_id and diplotype (eg: "AB" or "EE")
    pheno should have columns mouse_id (str), sex (M/F), DO_wave (1,2,...),
        and size_factor (float, DESeq2-style library size factors).
    """

    gene_totals = get_gene_totals(gene_id, gene_class_counts, diplotypes)

    pheno = pheno.join(
        gene_totals.select("mouse_id"),
        "mouse_id",
        maintain_order="right",
    )
    HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(gene_class_counts.haplotypes)}

    total_model = pm.Model(
        coords={
            "haplotypes": gene_class_counts.haplotypes,
            "samples": gene_class_counts.ids,
            "DOwaves": sorted(pheno["DOwave"].unique()),
        }
    )
    # Codings
    hap1 = gene_totals.select(
        pl.col("diplotype").str.slice(0, 1).replace_strict(HAP_TO_HAPNUM)
    )["diplotype"].to_numpy()
    hap2 = gene_totals.select(
        pl.col("diplotype").str.slice(1, 1).replace_strict(HAP_TO_HAPNUM)
    )["diplotype"].to_numpy()
    my_pheno = gene_totals.select("mouse_id").join(
        pheno, "mouse_id", how="left", maintain_order="left"
    )
    sex = my_pheno["sex"].replace_strict(SEX_TO_NUM).to_numpy()
    DOwave = my_pheno["DOwave"].cast(str).cast(int).to_numpy() - 1
    sf = np.log(my_pheno["size_factor"].to_numpy())
    mean_expr = (
        gene_totals["total_reads"] / my_pheno["size_factor"]
    ).to_numpy().mean() / 2
    with total_model:
        intercept = pm.Normal("intercept", mu=np.log(mean_expr), sigma=3)
        beta = pm.ZeroSumNormal("beta", sigma=2, dims="haplotypes")
        beta_M = pm.Normal("beta_M", sigma=2)
        beta_wave = pm.ZeroSumNormal("beta_wave", sigma=0.5, dims="DOwaves")
        log_disp = pm.Normal("log_disp", mu=np.log(0.01), sigma=2)
        mu = pm.Deterministic(
            "mu",
            (pm.math.exp(beta[hap1]) + pm.math.exp(beta[hap2]))
            * pm.math.exp(intercept + sex * beta_M + beta_wave[DOwave] + sf),
            dims="samples",
        )
        pm.NegativeBinomial(
            "total_reads",
            mu=mu,
            alpha=1 / np.exp(log_disp),
            observed=gene_totals["total_reads"].to_numpy(),
            dims="samples",
        )
    return total_model
