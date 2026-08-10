import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import lets_plot as lp
    import polars as pl
    import yaml

    return lp, mo, pl, yaml


@app.cell
def _(pl):
    counts = pl.read_parquet(
        "results/Adipose/Adipose.diploid.genes.founder_expected_read_counts.parquet"
    )
    return (counts,)


@app.cell
def _(counts):
    counts
    return


@app.cell
def _(pl):
    genotypes = pl.read_parquet("results/genotypes.parquet")
    return (genotypes,)


@app.cell
def _(genotypes):
    mouse_ids = sorted(genotypes["mouse_id"].unique())
    return (mouse_ids,)


@app.cell
def _(yaml):
    config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
    HAPLOTYPES = config["haplotypes"].split(",")
    return (HAPLOTYPES,)


@app.cell
def _(HAPLOTYPES, pl):
    def is_homozygous(col):
        return pl.col(col).is_in([f"{x}{x}" for x in HAPLOTYPES])

    return (is_homozygous,)


@app.cell
def _(HAPLOTYPES, counts, genotypes, is_homozygous, lp, pl):
    imbalance = counts.select(
        "gene_id",
        "mouse_id",
        "total",
        imbalance=pl.max_horizontal(
            [pl.col(hap) / pl.col("total") for hap in HAPLOTYPES]
        ),
    ).join(genotypes, ["gene_id", "mouse_id"])
    (
        lp.ggplot(
            imbalance.filter(
                pl.col("total") > 100,
                ~is_homozygous("genotype"),
            ),
            lp.aes(x="imbalance"),
        )
        + lp.geom_histogram()
    )
    return (imbalance,)


@app.cell
def _(imbalance):
    imbalance
    return


@app.cell
def _(mouse_ids, pl):
    def _():
        temp = []
        for mouse_id in mouse_ids:
            temp.append(
                pl.read_parquet(
                    f"processed/Adipose/gbrs_allele_unique_reads/{mouse_id}.allele_unique_reads.parquet"
                ).with_columns(mouse_id=pl.lit(mouse_id))
            )
        return pl.concat(temp)

    allele_unique = _()
    allele_unique
    return (allele_unique,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # QC checks
    """)
    return


@app.cell
def _(allele_unique, lp, mo, pl):
    # Check genotyping quality
    def _():
        df = allele_unique.group_by("mouse_id").agg(
            pl.col("total_reads").sum(),
            pl.col("diplotype_incompat_reads").sum(),
            pl.col("allele_specific_reads").sum(),
        )
        return lp.gggrid(
            [
                lp.ggplot(df, lp.aes("total_reads", "allele_specific_reads"))
                + lp.geom_point(tooltips=lp.layer_tooltips(["mouse_id"]))
                + lp.ylim(0),
                lp.ggplot(df, lp.aes("total_reads", "diplotype_incompat_reads"))
                + lp.geom_point(tooltips=lp.layer_tooltips(["mouse_id"]))
                + lp.ylim(0),
            ]
        ) + lp.ggsize(width=900, height=400)

    mo.vstack(
        [
            "We want allele-specific reads to be many and diplotype incompatible reads to be few. Any outlier samples could indicate a genotype file swap error.",
            _(),
        ]
    )
    return


@app.cell
def _(allele_unique, imbalance, is_homozygous, lp, mo, pl):
    # Check match of our allele-specific quants and GBRS quants
    # GBRS quants by EM should only approximate ours, though for gene-level will be pretty close
    # Multimappers between multiple genes will be the main exception, as will homozygous (which get 0 ASE in ours but 50%/50% in GBRS)
    def _():
        df = (
            allele_unique
            .select(
                "mouse_id",
                "gene_id",
                "allele_specific_reads",
                ASE_imbalance=(
                    pl.max_horizontal("haplotype_1_unique", "haplotype_2_unique")
                    / (pl.col("haplotype_1_unique") + pl.col("haplotype_2_unique"))
                ),
                ASE_total = "total_reads",
            )
            .join(
                imbalance.select(
                    "gene_id", "mouse_id", "genotype", "imbalance", "total"
                ),
                ["gene_id", "mouse_id"],
                how="inner",
            )
            .filter(~is_homozygous("genotype"), pl.col("total") > 100, pl.col("allele_specific_reads") > 30)
            .group_by("mouse_id")
            .agg(
                correlation = pl.corr("ASE_imbalance", "imbalance", method="spearman"),
                correlation_total = pl.corr("ASE_total", "total", method="spearman"),
            )
        )
        return lp.gggrid([
            lp.ggplot(df, lp.aes("mouse_id", "correlation_total")) + lp.geom_point() + lp.labs(y="corr(total reads)") + lp.ylim(-1,1),
            lp.ggplot(df, lp.aes("mouse_id", "correlation")) + lp.geom_point() + lp.labs(y="corr(imbalance)") + lp.ylim(-1,1),
        ])
    mo.vstack([
        "We expect GBRS quantified allele-specific imbalance and our own (non-EM) ASE quants to be close, at least when we have a large number of allele-specific reads",
        _()
    ])
    return


@app.cell
def _(allele_unique, lp, mo, pl):
    # Check by genotype
    def _():
        df = (
            allele_unique
                .with_columns(
                    hap1 = pl.col("diplotype").str.slice(0,1),
                    hap2 = pl.col("diplotype").str.slice(1,1),
                )
                .filter(pl.col("total_reads") > 10)
        )
        df = pl.concat([
            df.select(hap="hap1", unique="haplotype_1_unique", incompat="diplotype_incompat_reads"),
            df.select(hap="hap2", unique="haplotype_2_unique", incompat="diplotype_incompat_reads"),
        ]).with_columns(
            unique = pl.col("unique") + 1, # so that it works with log10() below
            incompat = pl.col("incompat") + 1, # so that it works with log10() below
        )
        hap = lp.as_discrete("hap", levels=list("ABCDEFGH"), order=1)
        return lp.gggrid([
            lp.ggplot(df.sample(n=10_000), lp.aes(hap, "unique", fill=hap)) + lp.geom_violin() + lp.scale_y_log10(),
            lp.ggplot(df.sample(n=10_000), lp.aes(hap, "incompat", fill=hap)) + lp.geom_violin() + lp.scale_y_log10(),
        ]) + lp.ggsize(width=900, height=400)
    def _grid():
        df = (
            allele_unique
                .with_columns(
                    hap1 = pl.col("diplotype").str.slice(0,1),
                    hap2 = pl.col("diplotype").str.slice(1,1),
                )
                .group_by("hap1", "hap2")
                .agg(N=pl.len())
        )
        hap1 = lp.as_discrete("hap1", levels=list("ABCDEFGH"), order=1)
        hap2 = lp.as_discrete("hap2", levels=list("ABCDEFGH"), order=-1)
        return lp.ggplot(df, lp.aes(hap1, hap2, fill="N")) + lp.geom_bin2d(stat="identity") + lp.scale_fill_viridis(option="magma", limits=[0])
    mo.vstack([
        "Check if any genotypes have different unqiue alignment properties, and the distribution of all diplotypes",
        _(),
        _grid(),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
