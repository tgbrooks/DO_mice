import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import lets_plot as lp
    import polars as pl
    import polars_bio as pb
    import statsmodels.api as sm
    import numpy as np
    import yaml

    return lp, mo, np, pb, pl, sm, yaml


@app.cell
def _():
    MIN_MEDIAN_EXPR_THRESHOLD = 50
    MIN_ASE_READS = 10
    return MIN_ASE_READS, MIN_MEDIAN_EXPR_THRESHOLD


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
    return HAPLOTYPES, config


@app.cell
def _(config, pb, pl):
    gene_annot = (
        pb.scan_gtf(
            config["gtf"], attr_fields=["gene_id", "gene_name", "gene_biotype"]
        )
        .filter(pl.col("type") == "gene")
        .collect()
    )
    return (gene_annot,)


@app.cell
def _(pl):
    # phenotypes were acquired from Dryad  and extracted from Rdata format:
    # https://datadryad.org/dataset/doi:10.5061/dryad.pj105
    pheno = pl.read_csv("phenotypes.csv.gz")
    pheno
    return


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


@app.cell(hide_code=True)
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
                lp.ggplot(
                    df, lp.aes("total_reads", "diplotype_incompat_reads")
                )
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


@app.cell(hide_code=True)
def _(allele_unique, imbalance, is_homozygous, lp, mo, pl):
    # Check match of our allele-specific quants and GBRS quants
    # GBRS quants by EM should only approximate ours, though for gene-level will be pretty close
    # Multimappers between multiple genes will be the main exception, as will homozygous (which get 0 ASE in ours but 50%/50% in GBRS)
    def _():
        df = (
            allele_unique.select(
                "mouse_id",
                "gene_id",
                "allele_specific_reads",
                ASE_imbalance=(
                    pl.max_horizontal(
                        "haplotype_1_unique", "haplotype_2_unique"
                    )
                    / (
                        pl.col("haplotype_1_unique")
                        + pl.col("haplotype_2_unique")
                    )
                ),
                ASE_total="total_reads",
            )
            .join(
                imbalance.select(
                    "gene_id", "mouse_id", "genotype", "imbalance", "total"
                ),
                ["gene_id", "mouse_id"],
                how="inner",
            )
            .filter(
                ~is_homozygous("genotype"),
                pl.col("total") > 100,
                pl.col("allele_specific_reads") > 30,
            )
            .group_by("mouse_id")
            .agg(
                correlation=pl.corr(
                    "ASE_imbalance", "imbalance", method="spearman"
                ),
                correlation_total=pl.corr(
                    "ASE_total", "total", method="spearman"
                ),
            )
        )
        return lp.gggrid(
            [
                lp.ggplot(df, lp.aes(y="correlation_total"))
                + lp.geom_boxplot()
                + lp.geom_jitter(
                    height=0, tooltips=lp.layer_tooltips(["mouse_id"])
                )
                + lp.labs(y="corr(total reads)"),
                lp.ggplot(df, lp.aes(y="correlation"))
                + lp.geom_boxplot()
                + lp.geom_jitter(
                    height=0, tooltips=lp.layer_tooltips(["mouse_id"])
                )
                + lp.labs(y="corr(imbalance)"),
            ]
        )

    mo.vstack(
        [
            "We expect GBRS quantified allele-specific imbalance and our own (non-EM) ASE quants to be close, at least when we have a large number of allele-specific reads",
            _(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(allele_unique, lp, mo, pl):
    # Check by genotype
    def _():
        df = allele_unique.with_columns(
            hap1=pl.col("diplotype").str.slice(0, 1),
            hap2=pl.col("diplotype").str.slice(1, 1),
        ).filter(pl.col("total_reads") > 10)
        df = pl.concat(
            [
                df.select(
                    hap="hap1",
                    unique="haplotype_1_unique",
                    incompat="diplotype_incompat_reads",
                ),
                df.select(
                    hap="hap2",
                    unique="haplotype_2_unique",
                    incompat="diplotype_incompat_reads",
                ),
            ]
        ).with_columns(
            unique=pl.col("unique") + 1,  # so that it works with log10() below
            incompat=pl.col("incompat")
            + 1,  # so that it works with log10() below
        )
        hap = lp.as_discrete("hap", levels=list("ABCDEFGH"), order=1)
        return lp.gggrid(
            [
                lp.ggplot(df.sample(n=10_000), lp.aes(hap, "unique", fill=hap))
                + lp.geom_violin()
                + lp.scale_y_log10(),
                lp.ggplot(
                    df.sample(n=10_000), lp.aes(hap, "incompat", fill=hap)
                )
                + lp.geom_violin()
                + lp.scale_y_log10(),
            ]
        ) + lp.ggsize(width=900, height=400)

    def _grid():
        au = allele_unique.with_columns(
            hap1=pl.col("diplotype").str.slice(0, 1),
            hap2=pl.col("diplotype").str.slice(1, 1),
        )
        df = au.group_by("hap1", "hap2").agg(N=pl.len())
        simple = (
            au.group_by("mouse_id")
            .agg(
                N_hom=pl.len().filter(pl.col("hap1") == pl.col("hap2")).sum(),
                N_het=pl.len().filter(pl.col("hap1") != pl.col("hap2")).sum(),
            )
            .with_columns(
                hom_het_fraction=pl.col("N_hom")
                / (pl.col("N_hom") + pl.col("N_het"))
            )
        )
        hap1 = lp.as_discrete("hap1", levels=list("ABCDEFGH"), order=1)
        hap2 = lp.as_discrete("hap2", levels=list("ABCDEFGH"), order=-1)
        founder_rates = (
            au["hap1"]
            .value_counts()
            .join(
                au["hap2"].value_counts().rename({"count": "hap2_count"}),
                left_on="hap1",
                right_on="hap2",
            )
            .select(hap="hap1", N=pl.col("count") + pl.col("hap2_count"))
            .with_columns(rate=pl.col("N") / pl.col("N").sum())
        )
        # expect ~1/8th of diplotypes to be homozygous but this depends on the exact
        # distribution of founder alleles in the population
        expected_hom_fraction = founder_rates.select(
            (pl.col("rate") * pl.col("rate")).sum()
        )["rate"][0]
        return lp.gggrid(
            [
                lp.ggplot(df, lp.aes(hap1, hap2, fill="N"))
                + lp.geom_bin2d(stat="identity")
                + lp.scale_fill_viridis(option="magma", limits=[0]),
                # lp.ggplot(simple, lp.aes("type", "N")) + lp.geom_boxplot() + lp.geom_jitter(height=0, tooltips=lp.layer_tooltips(["mouse_id"])) + lp.ylim(0),
                lp.ggplot(simple, lp.aes(y="hom_het_fraction"))
                + lp.geom_boxplot()
                + lp.geom_jitter(
                    height=0, tooltips=lp.layer_tooltips(["mouse_id"])
                )
                + lp.ylim(0)
                + lp.labs(y="fraction homozygous")
                + lp.geom_hline(yintercept=expected_hom_fraction, color="red"),
            ]
        ) + lp.ggsize(width=900, height=400)

    mo.vstack(
        [
            "Check if any genotypes have different unqiue alignment properties, and the distribution of all diplotypes. Check heterozygous versus homozygous rates: false heterozygous calls can be particularly bad for ASE. We expect 1/8 of diplotypes to be homozygous since 8 founders. Note that diplotypes are sorted so AB contains AB and BA options and we expect double the count for AB as for AA. Expected fraction homozygous shown in red line.",
            _(),
            _grid(),
        ]
    )
    return


@app.cell
def _(allele_unique, lp, mo, pl):
    MIN_TOTAL_READS = 10

    def _():
        frac = allele_unique.filter(
            pl.col("total_reads") > MIN_TOTAL_READS
        ).with_columns(
            frac=pl.col("allele_specific_reads") / pl.col("total_reads")
        )
        return (
            lp.ggplot(frac.sample(n=10000), lp.aes(x="frac"))
            + lp.geom_histogram()
            + lp.labs(x="fraction allele specific reads")
        )

    mo.vstack(
        [
            f"Assess what fraction of reads are informative (allele specific) across samples and genes. Samples with at least {MIN_TOTAL_READS} reads used.",
            _(),
        ]
    )
    return


@app.cell
def _(MIN_MEDIAN_EXPR_THRESHOLD, counts, lp, mo, np, pl):
    gene_expr_mat = counts.pivot("mouse_id", index="gene_id", values="total")
    _expr_mat = gene_expr_mat.drop("gene_id").to_numpy()
    _variance = (_expr_mat.std(axis=1) / (_expr_mat.mean(axis=1) + 1)) * (
        np.median(_expr_mat, axis=1) > MIN_MEDIAN_EXPR_THRESHOLD
    )
    high_variance_genes = np.argsort(-_variance)[
        :500
    ]  # use top 500 most variable genes

    def _():
        X = _expr_mat[high_variance_genes,]
        X = (X - np.mean(X, axis=1)[:, None]) / np.std(X, axis=1)[:, None]
        U, V, DT = np.linalg.svd(X, full_matrices=False)
        pca = pl.DataFrame(
            {
                "mouse_id": gene_expr_mat.columns[1:],
                "pca1": (U[:, [0]].T @ X).flatten(),
                "pca2": (U[:, [1]].T @ X).flatten(),
            }
        )
        return lp.ggplot(pca, lp.aes("pca1", "pca2")) + lp.geom_point(
            tooltips=lp.layer_tooltips(["mouse_id"])
        )

    mo.vstack(
        [
            "PCA plot of the samples",
            _(),
        ]
    )
    return (gene_expr_mat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gene-level analysis
    Here we just use gene counts, not separating out by alleles.
    We run DESeq2 which will give us dispersions that we can use for the rest of the analysis.
    DESeq2 does not allow predictor variables to vary across genes, so we just do the simplest model without genotype.
    """)
    return


@app.cell
def _(MIN_MEDIAN_EXPR_THRESHOLD, gene_expr_mat, mo, np, pl):
    import pydeseq2.dds

    genes_to_use = (
        np.median(gene_expr_mat.drop("gene_id").to_numpy(), axis=1)
        > MIN_MEDIAN_EXPR_THRESHOLD
    )

    @mo.persistent_cache
    def run_deseq2(gene_counts):
        X = gene_counts.drop("gene_id").to_numpy()[genes_to_use].T
        N_samples = gene_expr_mat.shape[1] - 1
        dds = pydeseq2.dds.DeseqDataSet(
            counts=X.astype(int),
            metadata=pl.DataFrame(
                {"intercept": np.ones(N_samples)}
            ).to_pandas(),
            design="~1",
        )
        dds.deseq2()
        return dds

    dds = run_deseq2(gene_expr_mat)
    return dds, genes_to_use


@app.cell
def _(dds, gene_annot, gene_expr_mat, genes_to_use, lp, mo, pl):
    gene_dispersions = pl.DataFrame(
        {
            "gene_id": gene_expr_mat.filter(genes_to_use)["gene_id"],
            "dispersion": dds.var["MAP_dispersions"],
            "normed_means": dds.var["_normed_means"],
        }
    )
    size_factors = pl.DataFrame(
        {
            "mouse_id": gene_expr_mat.drop("gene_id").columns,
            "size_factor": dds.obs["size_factors"],
        }
    )
    mo.vstack([
        "DESeq2 dispersion and mean estimates",
        lp.ggplot(
            gene_dispersions.join(gene_annot, "gene_id"),
            lp.aes(x="normed_means", y="dispersion"),
        )
        + lp.geom_pointdensity(
            tooltips=lp.layer_tooltips(["gene_id", "gene_name"])
        )
        + lp.scale_x_log10()
        + lp.scale_y_log10()
    ])
    return gene_dispersions, size_factors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modelling buffering from ASE
    """)
    return


@app.cell
def _(MIN_ASE_READS, allele_unique, is_homozygous, pl):
    # We need to discard samples that are sufficiently uninformative in terms of ASE
    to_use = allele_unique.with_columns(
        frac_AS=pl.col("allele_specific_reads") / pl.col("total_reads")
    ).filter(
        (pl.col("allele_specific_reads") > MIN_ASE_READS)
        | is_homozygous("diplotype"),
    )
    to_use
    return


@app.cell
def _(HAPLOTYPES, genotypes, pl):
    haplotype_counts = genotypes.with_columns(**{
        hap: pl.col("genotype").str.contains(hap).cast(int) + (pl.col("genotype") == f"{hap}{hap}").cast(int)
        for hap in HAPLOTYPES
    })
    return (haplotype_counts,)


@app.cell
def _(
    HAPLOTYPES,
    gene_dispersions,
    gene_expr_mat,
    genes_to_use,
    haplotype_counts,
    np,
    pl,
    size_factors,
    sm,
):
    gene_id = "ENSMUSG00000051747"
    assert sorted(gene_expr_mat.columns[1:]) == gene_expr_mat.columns[1:] # samples are sorted
    assert sorted(gene_expr_mat.columns[1:]) == size_factors['mouse_id'].to_list()
    disp = gene_dispersions.filter(gene_id=gene_id)['dispersion'][0]

    _selected_gene_ids = gene_expr_mat['gene_id'].filter(genes_to_use)
    results = []
    for gene_id in _selected_gene_ids:
        # Fit a model where expression is linear with haplotype counts for each haplotype
        endog = gene_expr_mat.filter(gene_id=gene_id).drop("gene_id").to_numpy().flatten()
        exog = haplotype_counts.filter(gene_id=gene_id).sort("mouse_id")
        hap_count = sm.GLM(
            endog = endog,
            exog = exog.select(HAPLOTYPES).to_numpy(),
            family = sm.families.NegativeBinomial(alpha=disp),
            offset = size_factors['size_factor'],
        ).fit()
        # Compare to model where all genotypes contribute equally
        _rmatrix = np.hstack([np.ones((7,1)), -np.eye(7)]) # A-B=0, A-C=0, ..., A-H=0
        diff_test = hap_count.f_test(_rmatrix)
    
        results.append({
            "gene_id": gene_id,
            "haplotype_effects": hap_count.params,
            "haplotype_se": hap_count.bse,
            "haplotype_effect_p": diff_test.pvalue,
        })
    results = pl.DataFrame(results)
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gene plots
    Interactive plots to visualize individual genes
    """)
    return


@app.cell
def _(gene_expr_mat, mo):
    gene_ids = sorted(gene_expr_mat['gene_id'])
    gene_selector = mo.ui.dropdown(gene_ids, searchable=True)
    gene_selector
    return (gene_selector,)


@app.cell
def _(HAPLOTYPES, allele_unique, gene_selector, lp, pl, size_factors):
    def _():
        HAPLOTYPE_COLORS = lp.scale_color_brewer(palette="Dark2").palette(len(HAPLOTYPES))
        au = (
            allele_unique.filter(gene_id = gene_selector.value)
                    .join(size_factors, "mouse_id")
        )
        max_expr = au.select(max=(pl.col("total_reads") / pl.col("size_factor")).max())['max'][0]
        plot_grid = []
        for hap1 in HAPLOTYPES:
            for hap2 in HAPLOTYPES:
                h1, h2 = sorted([hap1, hap2])
                hap_data = (
                    au
                    .filter(pl.col("diplotype") == f"{h1}{h2}")
                    .with_columns(
                        **{
                            h1: pl.col("haplotype_1_unique") / pl.col("size_factor"),
                             h2: pl.col("haplotype_2_unique") / pl.col("size_factor"),
                        },
                        unspecific = (pl.col("total_reads") - pl.col("allele_specific_reads")) / pl.col("size_factor"),
                    )
                    .unpivot(
                        on=[hap1, hap2, "unspecific"],
                        index="mouse_id",
                        variable_name="class",
                        value_name="expr",
                    )
                )
                _class = lp.as_discrete("class", levels=HAPLOTYPES+["unspecific"], order=1)
                plt = (
                    lp.ggplot(hap_data, lp.aes(x="mouse_id", y="expr", color=_class, fill=_class))
                    + lp.geom_bar(stat="identity")
                    + lp.theme_void()
                    + lp.ylim(0, max_expr)
                    + lp.theme(legend_position="none")
                    + lp.ggtitle(f"{hap1}{hap2}")
                    + lp.scale_color_manual(values=HAPLOTYPE_COLORS+["black"], breaks=HAPLOTYPES+["unspecific"])
                    + lp.scale_fill_manual(values=HAPLOTYPE_COLORS+["black"], breaks=HAPLOTYPES+["unspecific"])
                )
                plot_grid.append(plt)
        return lp.gggrid(plot_grid, ncol=len(HAPLOTYPES)) + lp.ggsize(900, 900)
    _()
    return


if __name__ == "__main__":
    app.run()
