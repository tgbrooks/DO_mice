import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assess simulated counts

    We simulated count-level data exhibiting buffering and then ran our models on those. Here, we check that the models perform as expected.
    """)
    return


@app.cell
def _():
    import lets_plot as lp
    import marimo as mo

    return lp, mo


@app.cell
def _():
    import pathlib
    import numpy as np
    import polars as pl
    import scipy.stats
    import scipy.integrate

    pl.Config(
        tbl_hide_dataframe_shape=True,
        tbl_rows=25,
        tbl_cols=20,
        tbl_hide_column_data_types=True,
    )

    temp = []
    for file in pathlib.Path("processed/simulated_counts/buffering").glob("*.txt"):
        temp.append(
            pl.read_csv(
                file,
                separator="\t",
                null_values="NA",
            )
        )
    results = pl.concat(temp)
    truth = pl.read_csv("processed/simulated_counts/true_params.txt", separator="\t")

    data = results.join(truth, "gene_id", suffix="_true").with_columns(
        type=pl.col("type").cast(pl.Enum(["no_cis", "no_buffering", "buffering"])),
        model=pl.col("model").cast(
            pl.Enum(["POISSON", "NEGATIVE_BINOMIAL", "SHARED_DISPERSION"])
        ),
    )

    haplotypes = list("ABCDEFGH")
    return data, haplotypes, np, pl, scipy, truth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Binomial model of ASE
    """)
    return


@app.cell(hide_code=True)
def _(data, haplotypes, lp, mo, pl):
    _est = data.unpivot([f"effect_{hap}" for hap in haplotypes], index=["type", "model", "gene_id", "true_buffering_factor"], variable_name="haplotype", value_name="est").with_columns(pl.col("haplotype").replace_strict({f"effect_{hap}": hap for hap in haplotypes}))
    _true = (
        data
            .with_columns(**{hap: pl.col(f"effect_{hap}_true") - pl.col(f"effect_H_true")
                        for hap in haplotypes if hap != "H"})
            .unpivot([hap for hap in haplotypes if hap != "H"], index=["type", "model", "gene_id", "true_buffering_factor"], variable_name="haplotype", value_name="true")
    )
    _data = _est.join(_true, ["type", "model", "gene_id", "haplotype", "true_buffering_factor"], how="inner")
    mo.vstack([
        "Check binomial models of allele-specific expression",
        (
            lp.ggplot(_data, lp.aes("true", "est", color="true_buffering_factor"))
            + lp.geom_point()
            + lp.theme_grey()
            + lp.scale_color_viridis(option="inferno", direction=-1)
            + lp.facet_grid(x="model", y="type")
            + lp.geom_abline(intercept=0, slope=1, color="red", linetype="dashed")
            + lp.coord_fixed()
            + lp.ggsize(800, 800)
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Negative binomial model of total counts
    """)
    return


@app.cell(hide_code=True)
def _(data, haplotypes, lp, mo, pl):
    _est = data.unpivot([f"total_{hap}" for hap in haplotypes], index=["type", "model", "gene_id", "true_buffering_factor"], variable_name="haplotype", value_name="est").with_columns(pl.col("haplotype").replace_strict({f"total_{hap}": hap for hap in haplotypes}))
    _true = (
        data
            .with_columns(**{hap: pl.col(f"effect_{hap}_true") - pl.col(f"effect_H_true")
                        for hap in haplotypes if hap != "H"})
            .unpivot([hap for hap in haplotypes if hap != "H"], index=["type", "model", "gene_id", "true_buffering_factor"], variable_name="haplotype", value_name="true")
    )
    _data = _est.join(_true, ["type", "model", "gene_id", "haplotype", "true_buffering_factor"], how="inner")
    mo.vstack([
        "Check negative binomial models of total read counts",
        (
            lp.ggplot(_data, lp.aes("true", "est", color="true_buffering_factor"))
            + lp.geom_point()
            + lp.theme_grey()
            + lp.scale_color_viridis(option="inferno", direction=-1)
            + lp.facet_grid(x="model", y="type")
            + lp.geom_abline(intercept=0, slope=1, color="red", linetype="dashed")
            + lp.coord_fixed()
            + lp.ggsize(800, 800)
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Buffering model
    """)
    return


@app.cell(hide_code=True)
def _(data, lp, mo, pl, truth):
    _data = data.join(truth, "gene_id").with_columns(
        true = pl.col("true_buffering_factor"),
        est = pl.col("buffering_factor"),
    )
    mo.vstack([
        "Check the buffering model",
        (
            lp.ggplot(_data, lp.aes("true", "est"))
            + lp.geom_point()
            + lp.geom_linerange(
                lp.aes(ymin="buffering_factor_ci_lo", ymax="buffering_factor_ci_hi"),
                color="red",
                alpha=1.0,
            )
            + lp.theme_grey()
            + lp.scale_color_viridis(option="inferno", direction=-1)
            + lp.facet_grid(x="model", y="type")
            + lp.geom_abline(intercept=0, slope=1, color="black", linetype="dashed")
            + lp.ggsize(800, 800)
        ),
    ])
    return


@app.cell(hide_code=True)
def _(data, lp, mo, np, pl, scipy):
    def _():
        print(
            "Check if the estimated buffering factor corresponds with the actual buffering effect"
        )
        print("  (Only for the buffering genes) ")
        temp = []
        aucs = dict()
        for (model,), _dat in data.group_by("model"):
            buff = _dat.filter(type="buffering")
            res = scipy.stats.linregress(
                buff["true_buffering_factor"], buff["buffering_factor"]
            )
            corr = np.corrcoef(buff["true_buffering_factor"], buff["buffering_factor"])[0, 1]

            ## AUC for identifying buffering
            roc = (
                _dat.sort("buffering_factor_ci_hi")
                .select(
                    statistic="buffering_factor_ci_hi",
                    n_below=pl.row_index() + 1,
                    n_true_positives=(pl.col("type") == "buffering").cum_sum(),
                    n_positives=(pl.col("type") == "buffering").sum(),
                )
                .with_columns(
                    fpr=(pl.col("n_below") - pl.col("n_true_positives"))
                    / (pl.len() - pl.col("n_positives")),
                    tpr=pl.col("n_true_positives") / pl.col("n_positives"),
                )
            )
            aucs[model] =  scipy.integrate.trapezoid(roc["tpr"], roc["fpr"])
            temp.append(roc.with_columns(
                model = pl.lit(model),
            ))
        df = pl.concat(temp)
        return mo.vstack([
            (
                lp.ggplot(
                    df,
                    lp.aes(x="fpr", y="tpr", color="model"),
                )
                + lp.geom_line()
                + lp.ggtitle("ROC")
            ),
            "AUC:",
            *[f"{model}: {auc:0.2f}" for model,auc in aucs.items()]
        ])
        


    _()
    return


if __name__ == "__main__":
    app.run()
