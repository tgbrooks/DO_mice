import polars as pl
import numpy as np
import pymc as pm
import arviz as az


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
            "DO_waves": sorted(pheno["DOwave"].unique()),
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
    DO_wave = my_pheno["DOwave"].cast(str).cast(int).to_numpy() - 1
    sf = np.log(my_pheno["size_factor"].to_numpy())
    mean_expr = (
        gene_totals["total_reads"] / my_pheno["size_factor"]
    ).to_numpy().mean() / 2
    with total_model:
        _hap1 = pm.Data("hap1", hap1, dims="samples")
        _hap2 = pm.Data("hap2", hap2, dims="samples")
        _totals = pm.Data(
            "totals", gene_totals["total_reads"].to_numpy(), dims="samples"
        )
        _mean_expr = pm.Data("mean_expr", mean_expr)

        intercept = pm.Normal("intercept", mu=np.log(_mean_expr), sigma=3)
        beta = pm.ZeroSumNormal("beta", sigma=2, dims="haplotypes")
        beta_male = pm.Normal("beta_male", sigma=2)
        beta_DO_wave = pm.ZeroSumNormal("beta_DO_wave", sigma=0.5, dims="DO_waves")
        log_disp = pm.Normal("log_disp", mu=np.log(0.01), sigma=2)
        mu = pm.Deterministic(
            "mu",
            (pm.math.exp(beta[_hap1]) + pm.math.exp(beta[_hap2]))
            * pm.math.exp(intercept + sex * beta_male + beta_DO_wave[DO_wave] + sf),
            dims="samples",
        )
        pm.NegativeBinomial(
            "total_reads",
            mu=mu,
            alpha=1 / np.exp(log_disp),
            observed=_totals,
            dims="samples",
        )
    return total_model


def _summary(idata, **kwargs):
    return pl.DataFrame(az.summary(idata, **kwargs).reset_index(names="variable"))


def summarize_total_model(idata):
    """Summarizes the posterior distribution from data sampled from an ASE model"""

    # Prioritize beta values since they're the values we actually care about
    beta_summary = _summary(idata, var_names=["beta"])
    # The rest are less important
    core_summary = _summary(
        idata, var_names=["intercept", "log_disp", "beta_DO_wave", "beta_male"]
    )
    # Sample-specific effects, less important
    rest_summary = _summary(idata, var_names=["mu"])

    return {
        "beta": beta_summary.rows_by_key("variable", unique=True, named=True),
        "core": core_summary.rows_by_key("variable", unique=True, named=True),
        "diagnostic": {
            "max_rhat_beta": beta_summary["r_hat"].max(),
            "max_rhat_core": core_summary["r_hat"].max(),
            "max_rhat_rest": rest_summary["r_hat"].max(),
            "min_ess_bulk_beta": beta_summary["ess_bulk"].min(),
            "min_ess_bulk_core": core_summary["ess_bulk"].min(),
            "min_ess_bulk_rest": rest_summary["ess_bulk"].min(),
            "min_ess_tail_beta": beta_summary["ess_tail"].min(),
            "min_ess_tail_core": core_summary["ess_tail"].min(),
            "min_ess_tail_rest": rest_summary["ess_tail"].min(),
        },
        "info": {
            "n_samples": idata["constant_data"]["totals"].shape[0],
            "n_diverging": int(idata["sample_stats"]["diverging"].sum()),
            "mean_tree_depth": float(idata["sample_stats"]["depth"].mean()),
            "maxdepth_reached_fraction": float(
                idata["sample_stats"]["maxdepth_reached"].mean()
            ),
            "sampling_time": idata["posterior"].attrs["sampling_time"],
        },
    }
