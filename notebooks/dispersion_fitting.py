import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import polars_bio as pb
    import numpy as np
    import scipy.sparse
    import scipy.optimize
    import scipy.stats
    import pymc as pm
    import arviz as az
    import time
    import xarray as xr

    import yaml

    from util.compressed_emase import load_compressed_emase
    from util.compat_classes import get_gene_class_counts, get_gene_totals, sparse_any
    from ase_buffering.ase_model import make_ase_model, summarize_read_classes, summarize_bias_rates, estimate_class_proportions, estimate_class_alpha
    from ase_buffering.ase_model_2steps import make_stage1_model, make_stage2_model, prepare_gene_data, stage1_likelihood
    from ase_buffering.total_counts_model import make_total_model


@app.cell
def _():
    config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
    HAPLOTYPES = config["haplotypes"].split(",")
    HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}
    SEX_TO_NUM = {"F": 0, "M": 1}
    return HAPLOTYPES, config


@app.cell
def _():
    import marimo as mo
    import lets_plot as lp

    return lp, mo


@app.cell
def _():
    size_factors = pl.read_csv("processed/Adipose/size_factors.txt", separator="\t")
    return


@app.cell
def _():
    effective_lengths = pl.read_parquet(
        f"processed/Adipose/effective_lengths.by_gene.parquet"
    )
    return (effective_lengths,)


@app.cell
def _(config):
    OUTLIER_MOUSE_IDS = config["outlier_ids"]
    return (OUTLIER_MOUSE_IDS,)


@app.cell
def _():
    gene_annot = (
        pb.scan_gtf(
            "gbrs_ref/v116/reference.gtf.gz",
            attr_fields=["gene_id", "gene_name", "gene_biotype"],
        )
        .filter(pl.col("type") == "gene")
        .collect()
    )
    return


@app.cell
def _():
    tx_annot = (
        pb.scan_gtf(
            "gbrs_ref/v116/reference.gtf.gz",
            attr_fields=["gene_id", "transcript_id"],
        )
        .filter(pl.col("type") == "transcript")
        .collect()
    )
    return (tx_annot,)


@app.cell
def _():
    genotypes = pl.read_parquet("processed/genotypes.parquet")
    allele_unique = pl.read_parquet("processed/Adipose/allele_unique_reads.parquet")
    return allele_unique, genotypes


@app.cell
def _(genotypes):
    genotypes2 = genotypes.select(
        "gene_id",
        "mouse_id",
        "genotype_confidence",
        diplotype="genotype",
    )

    return (genotypes2,)


@app.cell
def _(allele_unique):
    mouse_ids = sorted(allele_unique["mouse_id"].unique())
    return (mouse_ids,)


@app.cell
def _(HAPLOTYPES, OUTLIER_MOUSE_IDS, mouse_ids):
    # Load ALL the read compatibility data form all the mice
    all_compatibility_classes = {}
    for mouse_id in mouse_ids:
        if mouse_id in OUTLIER_MOUSE_IDS:
            continue
        all_compatibility_classes[mouse_id] = load_compressed_emase(
            f"processed/Adipose/gbrs/{mouse_id}.compressed.h5",
            [f"h{i}" for i in range(len(HAPLOTYPES))],
        )
    return (all_compatibility_classes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dispersion fitting

    Dispersion is given by $\alpha$ parameter such that: $var = \mu + \mu^2/\alpha$ (note: inverse of dispersion parameter used in some situations like DESeq2).
    We fit $\alpha$ as a function of $p$, the proportion of reads coming from a given class, by IRLS.
    """)
    return


@app.cell
def _(HAPLOTYPES, alpha_fit_df, lp, mo):
    mo.vstack([
        r"Per-diplotype variances of $p$, plotted against $p$. Fit trend line in magneta shows the variance predicted by the fit value of $\alpha$. Dashed black line shows the multinomial/Poisson-like variance baseline.",
        (
            lp.ggplot(
                alpha_fit_df.with_columns(
                    is_hom = pl.col("diplotype").is_in([f'{hap}{hap}' for hap in HAPLOTYPES]).cast(str).cast(pl.Enum(["false", "true"])),
                ).sample(fraction=1.0),# shuffle
                lp.aes("prop", "var", color="is_hom")
            )
            + lp.facet_wrap("gene_id", ncol=5)
            + lp.geom_point(lp.aes(alpha="in_fit"), tooltips=lp.layer_tooltips(["class_num", "n_samples", "diplotype", "q_hat_close"]))
            + lp.scale_x_log10 ()
            + lp.scale_y_continuous(trans="log10")
            + lp.labs(y="variance (of class proportions)")
            + lp.geom_line(lp.aes("prop", "fit_var"), data=alpha_fit_df.select("prop", "fit_var").drop_nulls(), color="magenta")
            #+ lp.geom_smooth(lp.aes("prop", "var"), inherit_aes=False, method="lm", deg=4, color="green")
            + lp.geom_line(lp.aes("prop", "poisson_var"), color="black", linetype="dashed")
        ),
    ])
    return


@app.cell
def _(alpha_fit_by_sample_df, lp, mo):
    mo.vstack([
        "Per-sample fit values of alpha.",
        (
            lp.ggplot(alpha_fit_by_sample_df.sample(10_000), lp.aes("prop", "alpha"))
            + lp.facet_wrap("gene_id")
            + lp.geom_point()
            + lp.labs(y="alpha (dispersion parameter)")
            + lp.scale_x_log10()
            + lp.scale_y_continuous(trans="log10")
        )
    ])
    return


@app.cell
def _(HAPLOTYPES, gene_ids, get_gene_data_cached):
    def fit_alphas(gene_id):
        gene_data, gene_class_counts = get_gene_data_cached(gene_id)
        hap1, hap2 = gene_data.hap1, gene_data.hap2
        _q_hat = estimate_class_proportions(gene_data.counts, hap1, hap2, len(HAPLOTYPES))
        alpha_fit, class_variance_df = estimate_class_alpha(gene_data)

        T = gene_data.counts.sum(axis=1).mean()
        p = pl.col("log_prop").exp()
        # Sum of p^2/alpha for just the classes other than this one
        other_term = (p**2 / pl.col("fit_alpha")).sum().over(["hap1", "hap2"])  - p**2 / pl.col("fit_alpha")
    
        _df = class_variance_df.with_columns(
            gene_id = pl.lit(gene_id),
            diplotype = np.array(HAPLOTYPES)[class_variance_df['hap1']] + np.array(HAPLOTYPES)[class_variance_df['hap2']],
            prop = p,
            poisson_var = pl.col("mean_inv_T") * p * (1-p),
            in_fit = pl.col("q_hat_close").cast(str).cast(pl.Enum(['false', 'true'])),
            # variance of x is mu + mu^2 / alpha,
            # variance of p is p(1-p)/T + p^2 [ (1-p^2)/alpha + sum_{k}p_k^2/alpha_k]
            # where k ranges over the other classes. Since p depends upon saple and class, this is hard to plot.
            # We plug in the mean T and average of the sum_{k} p_k^2/alpha_k term over all 
            fit_var = p*(1-p)/T + p**2 * ( (1-p)**2/pl.col("fit_alpha") + other_term),
            other_term = other_term,
        )

        _df_by_mouse = pl.DataFrame({
            "gene_id": gene_id,
            "mouse_id": gene_data.ids, 
            "alpha": alpha_fit,
            "total": gene_data.counts.sum(axis=1),
        }).join(pl.DataFrame({
            "mouse_id": gene_data.ids,
            "prop": (gene_data.q_hat[gene_data.hap1] + gene_data.q_hat[gene_data.hap2])/2,
        }), how="inner", on="mouse_id").explode(["prop", "alpha"], empty_as_null=False).with_columns(
            poisson_var = pl.col("prop")  / pl.col("total")
        )
        return [_df, _df_by_mouse]
    _dfs = []
    for _gene_id in gene_ids:
        try:
            _dfs.append(fit_alphas(_gene_id))
        except Exception as e:
            print(f"Skipping {_gene_id} due to error: {e}")
            continue

    alpha_fit_df = pl.concat([a for a,b in _dfs])
    alpha_fit_by_sample_df = pl.concat([b for a,b in _dfs])
    return alpha_fit_by_sample_df, alpha_fit_df


@app.cell
def _():
    gene_ids = [x.strip() for x in """
    ENSMUST00000078676
    ENSMUSG00000030738
    ENSMUSG00000041037
    ENSMUSG00000008575
    ENSMUSG00000021255
    ENSMUSG00000038539
    ENSMUSG00000025417
    ENSMUSG00000039457
    ENSMUSG00000046727
    ENSMUSG00000066235
    ENSMUSG00000037933
    ENSMUSG00000040811
    ENSMUSG00000024816
    ENSMUSG00000033111
    ENSMUSG00000121808
    ENSMUSG00000039891
    ENSMUSG00000001323
    ENSMUSG00000051934
    ENSMUSG00000051747
    ENSMUSG00000018286
    ENSMUSG00000038146
    ENSMUSG00000037185
    ENSMUSG00000032366
    ENSMUSG00000039419
    ENSMUSG00000009905
    """.strip().splitlines()]
    return (gene_ids,)


@app.cell
def _(all_compatibility_classes, mo, tx_annot):
    @mo.cache
    def get_gene_class_counts_cached(gene_id):
        return get_gene_class_counts(gene_id, tx_annot, all_compatibility_classes)

    return (get_gene_class_counts_cached,)


@app.cell
def _(effective_lengths, genotypes2, get_gene_class_counts_cached, mo):
    @mo.cache
    def get_gene_data_cached(gene_id):
        gene_class_counts =  get_gene_class_counts_cached(
                gene_id
        )
        gene_effective_lengths = effective_lengths.filter(gene_id=gene_id).select(
            "mouse_id", "haplotype", "effective_length"
        )
        gene_data = prepare_gene_data(
            gene_id,
            gene_class_counts,
            genotypes2,
            gene_effective_lengths,
            config=dict(
                min_genotype_confidence = 0.9,
            )
        )
    
        return gene_data, gene_class_counts

    return (get_gene_data_cached,)


@app.cell
def _(HAPLOTYPES):
    def diplotypes_to_hapnums(diplotypes):
        HAP_TO_HAPNUM = {hap:i for i,hap in enumerate(HAPLOTYPES)}
        hapnums = diplotypes.select(
            hap1=pl.col("diplotype").str.slice(0,1).replace_strict(HAP_TO_HAPNUM),
            hap2=pl.col("diplotype").str.slice(1,1).replace_strict(HAP_TO_HAPNUM),
        )
        return hapnums['hap1'].to_numpy(), hapnums['hap2'].to_numpy()

    return


@app.cell(disabled=True)
def _(idata_ase):
    summary = az.summary(idata_ase)
    return


if __name__ == "__main__":
    app.run()
