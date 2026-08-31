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
    import scipy.sparse

    return lp, mo, np, pb, pl, scipy, yaml


@app.cell
def _():
    MIN_MEDIAN_EXPR_THRESHOLD = 50
    MIN_ASE_READS = 10
    return (MIN_MEDIAN_EXPR_THRESHOLD,)


@app.cell
def _():
    MAX_INCOMPATIBLE_MICE = 3
    return (MAX_INCOMPATIBLE_MICE,)


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
def _(pl):
    size_factors = pl.read_csv("results/Adipose/size_factors.txt", separator="\t")
    return (size_factors,)


@app.cell
def _(genotypes):
    mouse_ids = sorted(genotypes["mouse_id"].unique())
    return (mouse_ids,)


@app.cell
def _(lp, yaml):
    config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
    HAPLOTYPES = config["haplotypes"].split(",")
    HAPLOTYPE_COLORS = lp.scale_color_brewer(palette="Dark2").palette( len(HAPLOTYPES) )
    OUTLIER_MOUSE_IDS = config['outlier_ids']
    return HAPLOTYPES, HAPLOTYPE_COLORS, OUTLIER_MOUSE_IDS, config


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
def _(config, pb, pl):
    tx_annot = (
        pb.scan_gtf(
            config["gtf"], attr_fields=["gene_id", "transcript_id"]
        )
        .filter(pl.col("type") == "transcript")
        .collect()
    )
    return (tx_annot,)


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
            ).sample(n=10000),
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
def _(OUTLIER_MOUSE_IDS, allele_unique, lp, mo, pl):
    # Check genotyping quality
    def _():
        df = allele_unique.group_by("mouse_id").agg(
            pl.col("total_reads").sum(),
            pl.col("diplotype_incompat_reads").sum(),
            pl.col("allele_specific_reads").sum(),
        ).with_columns(
            is_outlier = pl.col("mouse_id").is_in(OUTLIER_MOUSE_IDS)
        )
        return lp.gggrid(
            [
                lp.ggplot(df, lp.aes("total_reads", "allele_specific_reads", color="is_outlier"))
                + lp.geom_point(tooltips=lp.layer_tooltips(["mouse_id"]))
                + lp.scale_color_manual(
                    breaks = [False, True],
                    values = ["black", "red"],
                )
                + lp.ylim(0),
                lp.ggplot(
                    df, lp.aes("total_reads", "diplotype_incompat_reads", color="is_outlier")
                )
                + lp.geom_point(tooltips=lp.layer_tooltips(["mouse_id"]))
                + lp.scale_color_manual(
                    breaks = [False, True],
                    values = ["black", "red"],
                )
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
def _(OUTLIER_MOUSE_IDS, allele_unique, imbalance, is_homozygous, lp, mo, pl):
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
        ).with_columns(
            is_outlier = pl.col("mouse_id").is_in(OUTLIER_MOUSE_IDS)
        )
        return lp.gggrid(
            [
                lp.ggplot(df, lp.aes(y="correlation_total"))
                + lp.geom_boxplot(outlier_size=0)
                + lp.geom_jitter(
                    lp.aes(color="is_outlier"),
                    height=0, tooltips=lp.layer_tooltips(["mouse_id"])
                )
                + lp.scale_color_manual(
                    breaks = [False, True],
                    values = ["black", "red"],
                )
                + lp.labs(y="corr(total reads)"),
                lp.ggplot(df, lp.aes(y="correlation"))
                + lp.geom_boxplot(outlier_size=0)
                + lp.geom_jitter(
                    lp.aes(color="is_outlier"),
                    height=0, tooltips=lp.layer_tooltips(["mouse_id"])
                )
                + lp.scale_color_manual(
                    breaks = [False, True],
                    values = ["black", "red"],
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
def _(HAPLOTYPES, HAPLOTYPE_COLORS, allele_unique, lp, mo, pl):
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
                + lp.scale_fill_manual(
                    values=HAPLOTYPE_COLORS,
                    breaks=HAPLOTYPES,
                )
                + lp.scale_y_log10(),
                lp.ggplot(
                    df.sample(n=10_000), lp.aes(hap, "incompat", fill=hap)
                )
                + lp.geom_violin()
                + lp.scale_fill_manual(
                    values=HAPLOTYPE_COLORS,
                    breaks=HAPLOTYPES,
                )
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
                + lp.geom_boxplot(outlier_size=0)
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
def _(HAPLOTYPES, genotypes, pl):
    haplotype_counts = genotypes.with_columns(
        **{
            hap: pl.col("genotype").str.contains(hap).cast(int)
            + (pl.col("genotype") == f"{hap}{hap}").cast(int)
            for hap in HAPLOTYPES
        }
    )
    return (haplotype_counts,)


@app.cell
def _(HAPLOTYPES, haplotype_counts, mouse_ids):
    def _():
        mat = (
            haplotype_counts
                .unpivot(HAPLOTYPES, index=["gene_id", "mouse_id"], variable_name="haplotype")
                .pivot(index=["gene_id", "haplotype"], on="mouse_id", values="value")
        )
        return mat.select(mouse_ids).to_numpy()

    haplotype_mat = _()
    return (haplotype_mat,)


@app.cell(hide_code=True)
def _(
    MIN_MEDIAN_EXPR_THRESHOLD,
    OUTLIER_MOUSE_IDS,
    counts,
    haplotype_counts,
    haplotype_mat,
    lp,
    mo,
    mouse_ids,
    np,
    pl,
    scipy,
    size_factors,
):
    gene_expr_mat = counts.join(
        size_factors,
        "mouse_id",
    ).with_columns(
        norm_counts = pl.col("total") / pl.col("size_factor"),
    ).pivot("mouse_id", index="gene_id", values="norm_counts")
    _expr_mat = gene_expr_mat.drop("gene_id").to_numpy()
    _variance = (_expr_mat.std(axis=1) / (_expr_mat.mean(axis=1) + 1)) * (
        np.median(_expr_mat, axis=1) > MIN_MEDIAN_EXPR_THRESHOLD
    )
    high_variance_genes = np.argsort(-_variance)[
        :500
    ]  # use top 500 most variable genes

    def _():
        def run_pca(expr_mat, ids):
            X = expr_mat[high_variance_genes,]
            X = (X - np.mean(X, axis=1)[:, None]) / np.std(X, axis=1)[:, None]
            U, V, DT = scipy.sparse.linalg.svds(X, k=2)
            pca = pl.DataFrame(
                {
                    "mouse_id": ids,
                    "pca1": (U[:, [0]].T @ X).flatten(),
                    "pca2": (U[:, [1]].T @ X).flatten(),
                }
            )
            return pca
        pca = run_pca(_expr_mat, gene_expr_mat.columns[1:])
        ids_cleaned = [m for m in mouse_ids if m not in OUTLIER_MOUSE_IDS]
        _expr_mat_cleaned = gene_expr_mat.select(*ids_cleaned).to_numpy()
        pca_cleaned = run_pca(_expr_mat_cleaned, ids_cleaned)
        geno_mat = haplotype_counts
        pca_geno = run_pca(haplotype_mat, mouse_ids)
        genotype_pca = (
            lp.ggplot(pca_geno.with_columns(is_outlier = pl.col("mouse_id").is_in(OUTLIER_MOUSE_IDS)), lp.aes("pca1", "pca2", color="is_outlier"))
            + lp.geom_point(tooltips=lp.layer_tooltips(['mouse_id']))
            + lp.ggtitle("Genotype PCA")
            + lp.scale_color_manual(
                breaks = [False, True],
                values = ["black", "red"],
            )
        )
        return lp.gggrid([
            lp.ggplot(
                pca.with_columns(outlier = pl.col("mouse_id").is_in(OUTLIER_MOUSE_IDS)),
                lp.aes("pca1", "pca2", color="outlier")
            )
                + lp.geom_point(
                    tooltips=lp.layer_tooltips(["mouse_id"])
                )
                + lp.scale_color_manual(
                    breaks = [False, True],
                    values = ["black", "red"],
                )
                + lp.ggtitle("All samples")
            ,
            lp.ggplot(pca_cleaned, lp.aes("pca1", "pca2"))
                + lp.geom_point(tooltips=lp.layer_tooltips(["mouse_id"]))
                + lp.ggtitle("Outliers removed"),
        ]) + lp.ggsize(900,500), genotype_pca

    mo.vstack(
        [
            "PCA plot of the samples based off expression of high-variance genes and also on genotype.",
            *_(),
        ]
    )
    return (gene_expr_mat,)


@app.cell
def _(
    MAX_INCOMPATIBLE_MICE,
    OUTLIER_MOUSE_IDS,
    allele_unique,
    good_genes4,
    lp,
    np,
    pl,
):
    # Incompatible reads by gene
    num_incompat_by_gene = allele_unique.filter(
        ~pl.col("mouse_id").is_in(OUTLIER_MOUSE_IDS),
    ).group_by(
        "gene_id"
    ).agg(
        num_incompat = ((pl.col("diplotype_incompat_reads") / pl.col("total_reads")).fill_nan(0) > 0.05).sum()
    ).with_columns(
        good_gene = pl.col("gene_id").is_in(good_genes4),
    )
    _dat = num_incompat_by_gene.with_columns(
        pl.col("num_incompat").cut(np.linspace(0,250,31), include_breaks=True),
    ).group_by(
        ["num_incompat", "good_gene"]
    ).agg(
        num_genes = pl.len(),
    ).with_columns(
        num_incompat = pl.col("num_incompat").struct.field("breakpoint"),
    )

    lp.ggplot(_dat, lp.aes("num_incompat", "num_genes", fill="good_gene")) +  lp.geom_bar(stat="identity") + lp.scale_y_log10() + lp.geom_vline(xintercept=MAX_INCOMPATIBLE_MICE+0.5)
    return (num_incompat_by_gene,)


@app.cell
def _(MAX_INCOMPATIBLE_MICE, good_genes4, mo, num_incompat_by_gene, pl):
    good_genes5 = list(num_incompat_by_gene.filter(
        pl.col("gene_id").is_in(good_genes4),
        pl.col("num_incompat") <= MAX_INCOMPATIBLE_MICE,
    )['gene_id'])
    mo.vstack([f"Remaining: {len(good_genes5)} pass incompatible threshold"])
    return (good_genes5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Buffering results
    """)
    return


@app.cell
def _(pl):
    buffering_all = pl.read_csv(
        "results/Adipose/buffering.txt", separator="\t"
    )
    buffering_all
    return (buffering_all,)


@app.cell
def _(buffering_all, gene_annot, mo, pl):
    mo.vstack(
        [
            "Check how many failed convergence. After this step, we exclude those from our analysis.",
            buffering_all.select(
                frac_binom_failed=(
                    pl.col("convergence_code_binom") != 0
                ).mean(),
                frac_buffering_failed=(
                    pl.col("convergence_code_total") != 0
                ).mean(),
            ),
        ]
    )
    buffering = buffering_all.filter(
        pl.col("convergence_code_binom") == 0,
        pl.col("convergence_code_total") == 0,
    ).join(
        gene_annot.select("gene_id", "gene_name", "gene_biotype"),
        "gene_id",
    )
    return (buffering,)


@app.cell(hide_code=True)
def _(HAPLOTYPES, buffering, lp, mo, pl):
    genotype_effects = buffering.unpivot(
        [f"effect_{hap}" for hap in HAPLOTYPES],
        index=["gene_id", "anova_binom_p"],
        variable_name="haplotype",
        value_name="effect",
    ).with_columns(
        haplotype=pl.col("haplotype").str.strip_prefix("effect_"),
        abs_effect=pl.col("effect").abs(),
    )
    _hap = lp.as_discrete("haplotype", levels=HAPLOTYPES, order=1)
    mo.vstack(
        [
            "Plot genotype effects from the binomial model",
            lp.gggrid(
                [
                    lp.ggplot(
                        genotype_effects.sample(n=10_000),
                        lp.aes(x=_hap, y="abs_effect"),
                    )
                    + lp.geom_violin()
                    + lp.scale_y_log10()
                    + lp.ggtitle("All genes"),
                    lp.ggplot(
                        genotype_effects.filter(
                            pl.col("anova_binom_p") < 1e-3
                        ).sample(n=10_000),
                        lp.aes(x=_hap, y="abs_effect"),
                    )
                    + lp.geom_violin()
                    + lp.scale_y_log10()
                    + lp.ggtitle("Significant genes"),
                ]
            )
            + lp.ggsize(900, 500),
        ]
    )
    return


@app.cell
def _(buffering, good_genes5, lp, pl):
    (
        lp.ggplot(buffering.filter(pl.col('gene_id').is_in(good_genes5)), lp.aes("anova_binom_p", "buffering_factor"))
        + lp.scale_x_log10()
        + lp.geom_pointdensity(
            tooltips=lp.layer_tooltips(
                ["gene_id", "gene_name", "gene_biotype"]
            ),
            show_legend=False,
        )
        + lp.ggmarginal(sides="tr", layer=lp.geom_density())
        + lp.scale_color_viridis(option="magma")
    )
    return


@app.cell
def _(buffering, good_genes5, lp, mo, np, pl):
    _data = buffering.filter(
        pl.col("gene_id").is_in(good_genes5),
        pl.col("anova_binom_p") < 1e-25, # highly significant
    ).with_columns(
        pl.col("buffering_factor").cut(np.linspace(-0.25,1.25,31), include_breaks=True),
    ).with_columns(
        buffering_factor = pl.col("buffering_factor").struct.field("breakpoint"),
        is_significant = pl.col("buffering_factor_ci_hi") < 1.0,
    ).group_by(
        ["buffering_factor", "is_significant"]
    ).agg(
        num_genes = pl.len(),
    )
    mo.vstack([
        "Buffering factors in genes that are *highly* significant for having a genotype effect on allele-specific expression.",
        lp.ggplot(
            _data,
            lp.aes("buffering_factor", "num_genes", fill="is_significant")
        ) + lp.geom_bar(stat="identity")
    ])
    return


@app.cell
def _(buffering, good_genes5, lp, mo, np, pl):
    _data = buffering.filter(
        pl.col("gene_id").is_in(good_genes5),
        pl.col("anova_binom_p") < 1e-25, # highly significant
    ).with_columns(
        pl.col("deming_p_gof").cut(np.geomspace(1e-10,1,31), include_breaks=True)
    ).with_columns(
        deming_p_gof = pl.col("deming_p_gof").struct.field("breakpoint"),
    ).group_by(
        "deming_p_gof"
    ).agg(
        num_genes = pl.len()
    )
    mo.vstack([
        "Checking the goodness of fit tests for the deming models: testing whether there is a shared buffering factor common to all haplotypes.",
        lp.ggplot(
            _data,
            lp.aes("deming_p_gof", "num_genes"),
        ) + lp.geom_bar(stat="identity")
        + lp.scale_x_log10()
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gene plots
    Interactive plots to visualize individual genes
    """)
    return


@app.cell(hide_code=True)
def _(gene_annot, gene_expr_mat, mo, pl):
    _vals = (
        gene_expr_mat.select("gene_id")
        .unique()
        .join(gene_annot.select("gene_id", "gene_name"), "gene_id")
        .sort("gene_id")
        .drop_nulls()
    )
    _names = list(
        _vals.select(
            name=pl.col("gene_id") + " | " + pl.col("gene_name")
        ).drop_nulls()["name"]
    )
    gene_ids = list(_vals["gene_id"])
    _options = {_name: _id for _name, _id in zip(_names, gene_ids)}
    len(gene_ids)
    gene_selector = mo.ui.dropdown(_options, searchable=True)
    gene_selector
    return (gene_selector,)


@app.cell(hide_code=True)
def _(
    HAPLOTYPES,
    HAPLOTYPE_COLORS,
    allele_unique,
    gene_selector,
    lp,
    pl,
    size_factors,
):
    def _():
        au = (
            allele_unique.filter(gene_id=gene_selector.value)
            .join(size_factors, "mouse_id")
            .with_columns(
                hap1=pl.col("diplotype").str.slice(0, 1),
                hap2=pl.col("diplotype").str.slice(1, 2),
                norm_expr=pl.col("total_reads") / pl.col("size_factor"),
                norm_hap1=pl.col("haplotype_1_unique") / pl.col("size_factor"),
                norm_hap2=pl.col("haplotype_2_unique") / pl.col("size_factor"),
            )
        )
        by_hap_counts = pl.concat(
            [
                au.with_columns(
                    hap_count=pl.col("diplotype").str.count_matches(hap),
                    hap=pl.lit(hap),
                    hap_unique=pl.when(pl.col("hap1") == hap)
                    .then("norm_hap1")
                    .otherwise("norm_hap2"),
                )
                .with_columns(
                    hap_str=pl.when(pl.col("hap_count") == 2)
                    .then(pl.lit(f"{hap}{hap}"))
                    .otherwise(pl.lit(hap))
                )
                .filter(pl.col("hap_count") > 0)
                for hap in HAPLOTYPES
            ]
        )
        _hap = lp.as_discrete("hap", levels=HAPLOTYPES, order=1)
        median_expr = au["norm_expr"].median()
        plt_totals = (
            lp.ggplot(
                by_hap_counts, lp.aes(x="hap_str", y="norm_expr", color=_hap)
            )
            + lp.geom_boxplot(outlier_size=0, show_legend=False)
            + lp.geom_jitter(
                height=0, tooltips=lp.layer_tooltips(["mouse_id", "hap"])
            )
            + lp.geom_hline(
                yintercept=median_expr,
                color="black",
            )
            + lp.labs(
                y="total counts (normalized)",
                x="",
            )
            + lp.scale_color_manual(
                values=HAPLOTYPE_COLORS,
                breaks=HAPLOTYPES,
            )
            + lp.ylim(0)
        )
        _hap1 = lp.as_discrete("hap1", levels=HAPLOTYPES, order=1)
        _hap2 = lp.as_discrete("hap2", levels=HAPLOTYPES, order=1)
        df2 = pl.concat([
                au.select(
                    "mouse_id",
                    "hap1",
                    "hap2",
                    hap = "hap1",
                    norm_hap = "norm_hap1",
                ),
                au.select(
                    "mouse_id",
                    "hap1",
                    "hap2",
                    hap = "hap2",
                    norm_hap = "norm_hap2",
                ),
            ]).filter(pl.col("hap1") != pl.col("hap2"))
        df2 = pl.concat([df2, df2.with_columns(hap1="hap2", hap2="hap1")])
        plt_uniques = (
            lp.ggplot(df2,
                lp.aes(x=_hap2, y="norm_hap", color=_hap),
            )
            + lp.facet_wrap("hap1", scales="free_x", nrow=3)
            #+ lp.geom_boxplot(outlier_size=False)
            + lp.geom_jitter(
                height=0, tooltips=lp.layer_tooltips(["mouse_id"])
            )
            # + lp.geom_hline(
            #    yintercept = median_expr,
            #    color="black",
            # )
            + lp.scale_color_manual(
                values=HAPLOTYPE_COLORS,
                breaks=HAPLOTYPES,
            )
            + lp.labs(
                y="unique counts (normalized)",
                x="",
            )
            + lp.ylim(0)
        )
        return (
            lp.gggrid([plt_totals, plt_uniques], guides="collect")
            + lp.ggsize(900, 600)
            + lp.ggtitle(gene_selector.value)
        )

    _()
    return


@app.cell(hide_code=True)
def _(
    HAPLOTYPES,
    HAPLOTYPE_COLORS,
    OUTLIER_MOUSE_IDS,
    allele_unique,
    gene_selector,
    lp,
    pl,
    size_factors,
):
    def _():
        au = allele_unique.filter(gene_id=gene_selector.value).join(
            size_factors, "mouse_id"
        ).filter(
            ~pl.col("mouse_id").is_in(OUTLIER_MOUSE_IDS)
        )
        max_expr = au.select(
            max=(pl.col("total_reads") / pl.col("size_factor")).max()
        )["max"][0]
        plot_data = []
        for hap1 in HAPLOTYPES:
            for hap2 in HAPLOTYPES:
                h1, h2 = sorted([hap1, hap2])
                hap_data = (
                    au.filter(pl.col("diplotype") == f"{h1}{h2}")
                    .with_columns(
                        **{
                            h1: pl.col("haplotype_1_unique")
                            / pl.col("size_factor"),
                            h2: pl.col("haplotype_2_unique")
                            / pl.col("size_factor"),
                        },
                        incompatible = pl.col("diplotype_incompat_reads") / pl.col("size_factor"),
                        unspecific=(
                            pl.col("total_reads")
                            - pl.col("haplotype_1_unique")
                            - pl.col("haplotype_2_unique")
                            - pl.col("diplotype_incompat_reads")
                        )
                        / pl.col("size_factor"),
                        total = pl.col("total_reads") / pl.col("size_factor"),
                    )
                    .unpivot(
                        on=[hap1, hap2, "unspecific", "incompatible"],
                        index=["mouse_id", "total", "unspecific", "incompatible"],
                        variable_name="class",
                        value_name="expr",
                    )
                    .sort("mouse_id")
                    .with_columns(
                        mouse_idx = pl.col("mouse_id").is_first_distinct().cast(int).cum_sum(),
                    )
                )
                plot_data.append(hap_data.with_columns(hap1=pl.lit(hap1), hap2=pl.lit(hap2)))

        _class = lp.as_discrete(
            "class", levels=HAPLOTYPES + ["unspecific", "incompatible"], order=1
        )
        plt = (
            lp.ggplot(
                pl.concat(plot_data),
                lp.aes(
                    x="mouse_idx", y="expr", color=_class, fill=_class
                ),
            )
            + lp.facet_grid(x="hap1", y="hap2")
            + lp.geom_bar(
                stat="identity",
                tooltips=lp.layer_tooltips(["mouse_id"]).line("@class: @expr")
            )
            #+ lp.theme_void()
            + lp.theme_grey()
            + lp.ylim(0, max_expr)
            + lp.theme(tooltip_merge=True)
            + lp.scale_color_manual(
                values=HAPLOTYPE_COLORS + ["black", "red"],
                breaks=HAPLOTYPES + ["unspecific", "incompatible"],
            )
            + lp.scale_fill_manual(
                values=HAPLOTYPE_COLORS + ["black", "red"],
                breaks=HAPLOTYPES + ["unspecific", "incompatible"],
            )
        )
        return plt + lp.ggsize(1200,1200)
        #return lp.gggrid(plot_grid, ncol=len(HAPLOTYPES)) + lp.ggsize( 1800, 1800 )

    _()
    return


@app.cell
def _(HAPLOTYPES, HAPLOTYPE_COLORS, buffering, gene_selector, lp, mo, pl):
    def _():
        data = buffering.filter(gene_id = gene_selector.value)
        buff_factor = data['buffering_factor'][0]
        df = pl.DataFrame(dict(
            hap = [hap for hap in HAPLOTYPES if hap != "H"],
            x = data[[f'effect_{hap}' for hap in HAPLOTYPES if hap != 'H']].to_numpy()[0], # H is refernece, always 0
            x_se = data[[f'effect_{hap}_se' for hap in HAPLOTYPES if hap != 'H']].to_numpy()[0],
            y = data[[f'total_{hap}' for hap in HAPLOTYPES if hap != 'H']].to_numpy()[0],
            y_se = data[[f'total_{hap}_se' for hap in HAPLOTYPES if hap != 'H']].to_numpy()[0],
        )).with_columns(
            x_min = pl.col("x") - 1.96*pl.col("x_se"),
            x_max = pl.col("x") + 1.96*pl.col("x_se"),
            y_min = pl.col("y") - 1.96*pl.col("y_se"),
            y_max = pl.col("y") + 1.96*pl.col("y_se"),
        )
        xmin, xmax = min(df['x_min']), max(df['x_max'])
        ymin, ymax = min(df['y_min']), max(df['y_max'])
        lims = min(ymin,xmin), max(xmax, ymax)
        return (
            lp.ggplot(df, lp.aes("x", "y", color="hap"))
            + lp.geom_point(
                tooltips=lp.layer_tooltips(["hap"])
            )
            + lp.geom_errorbar(lp.aes(xmin="x_min", xmax="x_max"))
            + lp.geom_errorbar(lp.aes(ymin="y_min", ymax="y_max"))
            + lp.geom_point(data={"x": [0], "y": [0]}, mapping=lp.aes("x", "y"), alpha = 0, color="black") #ensure 0,0 is in view
            + lp.labs(x="beta (ASE, binomial GLM)", y="beta (total counts, NB GLM)")
            + lp.scale_color_manual(
                values=HAPLOTYPE_COLORS,
                breaks=HAPLOTYPES,
            )
            + lp.geom_abline(slope=buff_factor, intercept=0, color="red", linetype=2)
            + lp.geom_abline(slope=1, intercept=0, color="black", linetype=2)
            + lp.coord_fixed()
            + lp.xlim(*lims)
            + lp.ylim(*lims)
            + lp.ggtitle("Estimated effects by model type")
        )
    mo.vstack([
        "Plot the ASE (binomial) model fit parameters versus the total counts (NB GLM) fit parameters. In dash red, the buffering fit line and in black the reference diagonal line. Note that the buffering fit line takes into account covariance of the estimate, which is typically substantial and means that the fit may not line between the points (it's intercept is always 0).",
        _(),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Founder haplotype differences
    Investigate the differences between the transcriptomes of the different founders
    """)
    return


@app.cell(hide_code=True)
def _(pb, pl):
    # Assess the differences in transcriptomes
    transcript_lengths = (
        pb.scan_fasta("gbrs_ref/transcripts.fasta")
        .select(
            transcript_id=pl.col("name").str.split("_").list.get(0),
            haplotype=pl.col("name").str.split("_").list.get(1),
            seq_len=pl.col("sequence").str.len_bytes(),
        )
        .collect()
    )
    return (transcript_lengths,)


@app.cell(hide_code=True)
def _(lp, mo, pl, transcript_lengths):
    founder_length_differences = transcript_lengths.group_by(
        "transcript_id"
    ).agg(
        length_diff=pl.col("seq_len").max() - pl.col("seq_len").min(),
        length_ratio=pl.col("seq_len").max() / pl.col("seq_len").min(),
    )
    mo.vstack(
        [
            "Haplotypes can differ by indels and hence in their lengths. Here, we tally the number of transcripts by the difference in the longest and shortest haplotypes.",
            lp.ggplot(
                founder_length_differences.with_columns(
                    pl.col("length_diff").cut(
                        [0, 1, 5, 10, 25, 100, 250, 500]
                    ),
                )
                .group_by("length_diff")
                .agg(num_transcripts=pl.len()),
                lp.aes("length_diff", "num_transcripts"),
            )
            + lp.geom_bar(stat="identity"),
        ]
    )
    return (founder_length_differences,)


@app.cell
def _(founder_length_differences, pl, tx_annot):
    gene_founder_length_differences = founder_length_differences.join(
        tx_annot,
        "transcript_id",
    ).group_by("gene_id").agg(
        pl.col("length_diff").max(),
        pl.col("length_ratio").max()
    )
    return (gene_founder_length_differences,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simulated reads
    We generated simulated, perfet reads from every gene in order to test what the 'ideal' ratios of allele-unique reads would be.
    Reads were generated from each founder haplotype and then were considered as if having come from each possible diplotype containing that founder haplotype.

    This is used to select a set of 'good' genes for further use based off their behavior in simulated reads.
    """)
    return


@app.cell
def _(pl):
    sim_reads = pl.read_parquet(
        "processed/simulated_reads/allele_unique_reads.parquet"
    ).filter(
        pl.col("source_haplotype") != pl.col("other_haplotype")
    )  # homozygous diplotypes provide no information
    return (sim_reads,)


@app.cell
def _(mo, pl, sim_reads):
    problematic_genes = (
        sim_reads
            .filter(pl.col("total_reads") > 0)
            .group_by("gene_id")
            .agg(
                any_problems = (pl.col("haplotype_2_unique") > 0).any() | (pl.col("diplotype_incompat_reads").any() > 0),
            )
    )
    good_genes = sorted(list(problematic_genes.filter(~pl.col("any_problems"))['gene_id']))
    mo.vstack([
        "We consider genes 'problematic' if expression from one haplotype gets assigned as unique expression of another haplotype. This probably happens due to gene-level multimapping. Fortunately, this is a minority of genes. And we'll exclude them from here.",
        problematic_genes['any_problems'].value_counts(name="num genes")
    ])
    return (good_genes,)


@app.cell(hide_code=True)
def _(gene_founder_length_differences, good_genes, lp, mo, pl, sim_reads):
    sim_totals = sim_reads.pivot(
        "diplotype", index="gene_id", values="total_reads"
    )
    sim_total_diffs = (
        sim_reads.group_by("gene_id")
        .agg(
            pl.col("total_reads").min().alias("min_total"),
            pl.col("total_reads").max().alias("max_total"),
        )
        .filter(pl.col("max_total") > 0)
        .filter(pl.col('gene_id').is_in(good_genes))
        .with_columns(
            total_ratio = pl.col('max_total') / pl.col("min_total"),
        )
        .join(
           gene_founder_length_differences,
            "gene_id",
        )
    )
    mo.vstack(
        [
            """First, check whether all the *total* reads are the same.
        This could vary slightly if founders differ by indels or if multimapping to other genes is affected""",
            (
                lp.ggplot(sim_total_diffs, lp.aes("min_total", "max_total"))
                + lp.geom_pointdensity(
                    tooltips=lp.layer_tooltips(["gene_id"])
                )
                + lp.geom_abline(intercept=0, slope=1)
                + lp.scale_x_log10()
                + lp.scale_y_log10()
            ),
            (
                lp.ggplot(sim_total_diffs, lp.aes("length_ratio", "total_ratio"))
                + lp.geom_pointdensity(
                    tooltips=lp.layer_tooltips(["gene_id"])
                )
                + lp.geom_abline(intercept=0, slope=1)
                + lp.scale_x_log10()
                + lp.scale_y_log10()
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(good_genes, mo, pl, sim_reads):
    MAX_TOTAL_READ_DISCREPANCY = 20
    problematic_genes2 = (
        sim_reads
            .filter(pl.col("total_reads") > 0)
            .filter(pl.col("gene_id").is_in(good_genes))
            .group_by("gene_id")
            .agg(
                any_total_problems = (pl.col("total_reads").max() - pl.col('total_reads').min() > MAX_TOTAL_READ_DISCREPANCY).any()
            )
    )
    good_genes2 = sorted(list(problematic_genes2.filter(~pl.col("any_total_problems"))['gene_id']))
    mo.vstack([
        f"Since it looks like gene lengths aren't the major difference in totals, these likely come down to multimapping from other loci. Therefore, we further exclude genes that have problems with their total counts. Specifically, check that they never differ by more than {MAX_TOTAL_READ_DISCREPANCY} counts between haplotypes.",
        problematic_genes2['any_total_problems'].value_counts(name="num genes"),
    ])
    return (good_genes2,)


@app.cell
def _(HAPLOTYPES, good_genes2, lp, mo, np, pl, sim_reads):
    # Also check whether good genes have good allelic ratios
    MAX_UNIQUE_READS_RATIO = 1.05
    def _():
        good = sim_reads.filter(pl.col('gene_id').is_in(good_genes2))
        temp = []
        for hap1 in HAPLOTYPES:
            for hap2 in HAPLOTYPES:
                if hap1 >= hap2:
                    continue
                AB = good.filter(source_haplotype = hap1, other_haplotype = hap2)
                BA = good.filter(source_haplotype = hap2, other_haplotype = hap1)
                au_ratio = (
                    AB.join(BA, "gene_id")
                    .select(
                        "gene_id",
                        hap1 = pl.lit(hap1),
                        hap2 = pl.lit(hap2),
                        A = pl.col("haplotype_1_unique"),
                        B = pl.col("haplotype_1_unique_right"),
                        au_ratio = pl.when((pl.col("haplotype_1_unique") == 0) & (pl.col("haplotype_2_unique") == 0))
                            .then(pl.lit(1))
                            .otherwise(
                                pl.col("haplotype_1_unique") / pl.col("haplotype_1_unique_right")
                            ),
                    )
                )
                temp.append(au_ratio)
        return pl.concat(temp)
    sim_au_ratios = _().group_by("gene_id").agg(pl.col("au_ratio").log().abs().max().exp())
    _cutoffs = np.linspace(1, 1.20, 101)
    _by_cutoff = sim_au_ratios.join(
        pl.DataFrame({"ratio_cutoff": _cutoffs}),
        how="cross",
    ).filter(pl.col("ratio_cutoff") > pl.col("au_ratio")).group_by("ratio_cutoff").agg(num_genes_below=pl.len())
    mo.vstack([
        f"Different source haplotypes and diplotypes can create different ratios of unique reads. Here, we filter again to just those with at most {MAX_UNIQUE_READS_RATIO} ratio between highest and lowest number of unique reads across the different haplotypes.",
        lp.ggplot(_by_cutoff, lp.aes(x="ratio_cutoff", y="num_genes_below")) + lp.geom_line() + lp.geom_vline(xintercept=MAX_UNIQUE_READS_RATIO, linetype=3) + lp.ylim(0)
    ])
    return MAX_UNIQUE_READS_RATIO, sim_au_ratios


@app.cell
def _(MAX_UNIQUE_READS_RATIO, pl, sim_au_ratios):
    good_genes3 = sorted(sim_au_ratios.filter(pl.col("au_ratio") < MAX_UNIQUE_READS_RATIO)['gene_id'])
    return (good_genes3,)


@app.cell
def _(good_genes3, lp, mo, pl, sim_reads, sim_source_counts):
    MAX_OVERMAPPING_RATIO = 1.05
    _data = sim_reads.join(sim_source_counts.select("gene_id", source_haplotype="haplotype", expected_reads="num_reads"), ["gene_id","source_haplotype"],).filter(pl.col("gene_id").is_in(good_genes3))
    good_genes4 = list(
        _data
        .group_by("gene_id")
        .agg(ratio = (pl.col("total_reads") / pl.col("expected_reads")).min())
        .filter(pl.col('ratio') < MAX_OVERMAPPING_RATIO)['gene_id'].unique()
    )
    mo.vstack([
        f"Check for multimapping between genes by comparing the number of reads that were generated from a gene to the number actually mapping there. Discard any genes with more than {MAX_OVERMAPPING_RATIO} the expected value",
        lp.ggplot(_data.sample(n=5_000), lp.aes("expected_reads", "total_reads"))
        + lp.geom_pointdensity()
        + lp.scale_color_viridis(option="magma")
        + lp.scale_x_log10()
        + lp.scale_y_log10()
    ])
    return (good_genes4,)


@app.cell
def _(pl):
    sim_source_counts = pl.read_csv("results/simulated_reads/source_counts_by_gene.txt", separator="\t")
    return (sim_source_counts,)


@app.cell
def _(good_genes4, mo):
    mo.vstack([f"Final total of genes selected for use: {len(good_genes4)}"])
    return


if __name__ == "__main__":
    app.run()
