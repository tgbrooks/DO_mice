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
    import json
    import scipy.sparse

    return json, lp, mo, np, pb, pl, scipy, yaml


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
    genotypes = pl.read_parquet("processed/genotypes.parquet")
    return (genotypes,)


@app.cell
def _(pl):
    size_factors = pl.read_csv("processed/Adipose/size_factors.txt", separator="\t")
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
    return HAPLOTYPES, HAPLOTYPE_COLORS, OUTLIER_MOUSE_IDS


@app.cell
def _(pb, pl):
    gene_annot = (
        pb.scan_gtf(
            "gbrs_ref/v116/reference.gtf.gz", attr_fields=["gene_id", "gene_name", "gene_biotype"]
        )
        .filter(pl.col("type") == "gene")
        .collect()
    )
    return (gene_annot,)


@app.cell
def _(pb, pl):
    tx_annot = (
        pb.scan_gtf(
            "gbrs_ref/v116/reference.gtf.gz", attr_fields=["gene_id", "transcript_id"]
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
    return (pheno,)


@app.cell
def _(HAPLOTYPES, pl):
    def is_homozygous(col):
        return pl.col(col).is_in([f"{x}{x}" for x in HAPLOTYPES])

    return


@app.cell
def _(pl):
    allele_unique = pl.read_parquet("processed/Adipose/allele_unique_reads.parquet")
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
    allele_unique,
    haplotype_counts,
    haplotype_mat,
    lp,
    mo,
    mouse_ids,
    np,
    pheno,
    pl,
    scipy,
    size_factors,
):
    gene_expr_mat = allele_unique.join(
        size_factors,
        "mouse_id",
    ).with_columns(
        norm_counts = pl.col("total_reads") / pl.col("size_factor"),
    ).pivot("mouse_id", index="gene_id", values="norm_counts")
    _expr_mat = gene_expr_mat.drop("gene_id").to_numpy()
    _variance = (_expr_mat.std(axis=1) / (_expr_mat.mean(axis=1) + 1)) * (
        np.median(_expr_mat, axis=1) > MIN_MEDIAN_EXPR_THRESHOLD
    )
    high_variance_genes = np.argsort(-_variance)[
        :500
    ]  # use top 500 most variable genes

    def normalize_mat(expr_mat):
        X = expr_mat[high_variance_genes,]
        X = (X - np.mean(X, axis=1)[:, None]) / np.std(X, axis=1)[:, None]
        return X

    def run_pca(expr_mat, ids):
        X = normalize_mat(expr_mat)
        U, V, DT = scipy.sparse.linalg.svds(X, k=2)
        pca = pl.DataFrame(
            {
                "mouse_id": ids,
                "pca1": (U[:, [0]].T @ X).flatten(),
                "pca2": (U[:, [1]].T @ X).flatten(),
            }
        )
        return pca

    def _():
        pca = run_pca(_expr_mat, gene_expr_mat.columns[1:])
        ids_cleaned = [m for m in mouse_ids if m not in OUTLIER_MOUSE_IDS]
        _expr_mat_cleaned = gene_expr_mat.select(*ids_cleaned).to_numpy()
        pca_cleaned = run_pca(_expr_mat_cleaned, ids_cleaned)
        geno_mat = haplotype_counts
        pca_geno = run_pca(haplotype_mat, mouse_ids)
        genotype_pca = (
            lp.ggplot(
                pca_geno.with_columns(is_outlier = pl.col("mouse_id").is_in(OUTLIER_MOUSE_IDS))
                    .join(pheno, left_on="mouse_id", right_on="mouse.id"),
                lp.aes("pca1", "pca2", color="is_outlier")
            )
            + lp.geom_point(tooltips=lp.layer_tooltips(['mouse_id', "sex", "DOwave"]))
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
            lp.ggplot(
                pca_cleaned.join(pheno, left_on="mouse_id", right_on="mouse.id"),
                lp.aes("pca1", "pca2", color="sex")
            )
                + lp.geom_point(tooltips=lp.layer_tooltips(["mouse_id"]))
                + lp.ggtitle("Outliers removed")
                + lp.scale_color_manual(breaks=["F", "M"], values=["orange", "green"]),
        ]) + lp.ggsize(900,500), genotype_pca

    mo.vstack(
        [
            "PCA plot of the samples based off expression of high-variance genes and also on genotype.",
            *_(),
        ]
    )
    return gene_expr_mat, normalize_mat


@app.cell
def _(gene_expr_mat, lp, mo, normalize_mat, np):
    # Check for duplicate samples
    _expr_mat = gene_expr_mat.drop("gene_id").to_numpy()
    _X = normalize_mat(_expr_mat)
    sample_correlations = np.corrcoef(_X.T)
    _corrs = (sample_correlations - np.eye(sample_correlations.shape[0])).flatten()
    mo.vstack([
        "Check for duplicate samples with very high correlation:",
        lp.ggplot({"corr": _corrs}, lp.aes(y="corr")) + lp.geom_boxplot()
        + lp.labs(x='', y="sample-sample correlation")
    ])
    return


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


@app.cell
def _(genotypes, lp, mo, pl):
    # Check genotyping confidence
    geno_summary = (
        genotypes
            .group_by("gene_id")
            .agg(
                over_095 = (pl.col("genotype_confidence") < 0.95).sum(),
                over_09 = (pl.col("genotype_confidence") < 0.9).sum(),
                over_08 = (pl.col("genotype_confidence") < 0.8).sum(),
            )
            .unpivot(on=["over_095", "over_09", "over_08"], index="gene_id", value_name="n_samples_failing", variable_name="threshold")
    )
    mo.vstack([
        "Genotyping calls are made at multiple markers and involves uncertainty. We defined 'confidence' of the call by averaging over all markers overlapping the gene and one up and downstream of it. 1 confidence indicates all markers completely agree with the assigned genotype, 0 means complete disagreement. Here we investigate what fraction of samples would pass various thresholds.",
        (
             lp.ggplot(geno_summary, lp.aes("n_samples_failing", color="threshold", fill="threshold"))
                + lp.stat_ecdf()
                + lp.labs(x = "number samples failing", y="fraction of genes")
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Buffering results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We build two models and compare them to get the buffering rate.

    ### Binomial model of ASE:

    First, allele-specific expression is modelled using a binomial GLM.
    We use a matrix of 7 predictors $X_{binom}$ built by putting a 1 in the column corresponding to the first haplotype
    and a -1 in the column corresponding to the second haplotype (ignoring all H haplotypes since those are used as the reference).
    If a sample is homozygous, we drop it.
    No intercept column is included (50/50 expression is the default).
    Then the outcome variable is the binomial result `(n_successes, n_failures)` where `n_successes` is the number of reads uniquely mapping to the first haplotype (A) and `n_failures` is the number of reads uniquely mapping to the second haplotype (B).

    ```
    (hap1_unique, hap2_unique) ~ Binom( expit(X_binom beta_binom), n = hap1_unique + hap2_unique )
    ```

    ### Negative binomial model of total read counts:

    Next, we use a negative binomial GLM to model the total read counts to a gene (summing over both alleles as well as all allele-ambiguous reads).
    Here, the predictor matrix $X_{nb}$ is just the dosage of each of 7 haplotypes (again, using H as the reference).
    Unlike the binomial model, we also include an intercept, covariates (sex and DO wave) as well as a random effect for kinship: since the binomial model always compares within the same mouse, covariates would cancel out.
    Library size normalization factors are fit by DESeq2 first, which are likewise included only in the negative binomial model.

    ```
    total_counts ~ NBinom(exp(X_nb beta_nb + covariates))
    ```

    ### Connecting the models

    Suppose for now that expression is actually simpler: just Poisson with each allele independent.
    The (unknown) true total reads to a haplotype $i$ is $y_i$ of which $u_i$ are uniquely mapping to that haplotype.
    We assume:
    $$
    y_i \sim \mbox{Poisson}(\lambda_i) \\
    u_i \sim \mbox{Binom}(y_i, p)
    $$
    Note that we assume $p$, the fraction of reads that are unique, doesn't depend upon the haplotype.
    Then
    $$
    u_i \sim \mbox{Poisson}(p \lambda _i)
    $$
    and the conditional distribution fo the unique reads is:
    $$
    u_i | u_1 + u_2 \sim \mbox{Binom}(u_1 + u_2, \quad \lambda_i/(\lambda_1 + \lambda_2))
    $$
    see [Wikipedia](https://en.wikipedia.org/wiki/Poisson_distribution#Sums_of_Poisson-distributed_random_variables:~:text=are%20independent%2C%20then%20the%20distribution%20of).
    In particular, $\mbox{E}[u_1/(u_1 + u_2)] = \lambda_1 / (\lambda_1 + \lambda_2)$.

    The parameter $\beta_{binom}$ from the binomial GLM with a logistic link comes from:
    $$
    \begin{align}
    \mbox{E}[u_1 / (u_1 + u_2)]
        &= \mbox{expit}(\beta_{binom, 1} - \beta_{binom, 2}) \\
        &= \exp (\beta_{binom, 1} - \beta_{binom, 2}) / (\exp(\beta_{binom,1} - \beta_{binom,2}) + 1) \\
        &= \exp (\beta_{binom, 1}) / (\exp(\beta_{binom,1}) + \exp(\beta_{binom, 2})) \\
    \end{align}
    $$
    By the previous identity, $\exp(\beta_{binom,i}) = \lambda_i$.

    Meanwhile, the negative binomial GLM with log link gives:
    $$
    \begin{align}
    \mbox{E}[y_1 + y_2] &= \exp(\beta_{nb, 1} + \beta_{nb, 2}) \\
        &\approx \exp(\beta_{nb,1}) + \exp(\beta_{nb, 2})
    \end{align}
    $$
    for relatively small haplotype effects.
    So $\exp(\beta_{nb,i}) \approx \lambda_i$.

    ### With buffering
    All this so far assumes no buffering.
    If instead we have for heterozygous:
    $$
    y_i \sim \mbox{Poisson}(\lambda_i (\lambda_*/ (\lambda_1 + \lambda_2))^{1-bf})
    $$
    and for homozygous:
    $$
    y_i \sim \mbox{Poisson}(\lambda_i (\lambda_*/ (2\lambda_i))^{1-bf}).
    $$
    where $\lambda_*$ is the 'target' mean and $bf$ is a buffering-factor:
    $bf$ of 0 corresponds to the total expression $\lambda_1 + \lambda_2$ being shrunk completely towards $\lambda_*$ and $bf=1$ corresponds to no buffering.

    Nothing changes in the binomial model since the ratio $\mbox{E}[u_1/(u_1+u_2)]$ is unchanged.

    However, in the negative binomial model, we now have for heterozygous
    $$
    \begin{align}
    \mbox{E}[y_1 + y_2]
        &= (\lambda_1 + \lambda_2) (\lambda_* / (\lambda_1 + \lambda))^{1 - bf} \\
        &= (\lambda_1 + \lambda_2)^{bf} (\lambda_*)^{1 - bf}
    \end{align}
    $$
    and for homozygous:
    $$
    \begin{align}
    \mbox{E}[y_i]
        &= 2 \lambda_i (\lambda_* / (2 \lambda_1))^{1 - bf} \\
        &= (2 \lambda_i)^{bf} (\lambda_*)^{1-bf}
    \end{align}
    $$
    So the NB GLM satisfies:
    $$
    \beta_{nb, i} \approx bf \log(\lambda_i)
    $$
    with the intercept term absorbing the $\lambda_*^{1-bf}$ part.
    This approximation again depends upon the assumption that effect sizes are small.

    Now we see that the buffering factor is estimated by the ratio:
    $$
    \widehat{bf} \approx \beta_{nb,i} / \beta_{binom, i}.
    $$

    Since we have 8 founder types, we have 7 estimates and we instead esimate $bf$ by a Deming-like regression of $\beta_{binom, i}$ versus $\beta_{nb,i}$, accounting for the known estimation variance from each model.
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


@app.cell
def _(buffering, good_genes5, lp, np, pl):
    def moving_median(x,y, eval_pts, sd):
        x = np.asarray(x)
        y = np.asarray(y)
        weights = np.exp(-(x[None,:]-eval_pts[:,None])**2/(2*sd**2))
        weights = weights / np.sum(weights, axis=1)[:,None]
        order = np.argsort(y)
        median_idx = np.empty(eval_pts.shape, dtype=np.int64)
        for i, w in enumerate(weights):
            median_idx[i] = np.searchsorted(
                np.cumsum(w[order]),
                0.5
            )
        median_idx = np.minimum(y.shape[0]-1, median_idx)
        medians = y[order][median_idx]
        return medians
    _dat = buffering.filter(
        pl.col('gene_id').is_in(good_genes5),
    ).with_columns(
        -pl.col("anova_binom_p").log10().alias("-log10_anova_binom_p")
    )

    _eval_pts = np.linspace(_dat['-log10_anova_binom_p'].min(), _dat['-log10_anova_binom_p'].max(), 100)
    _mm = moving_median(
        _dat['-log10_anova_binom_p'],
        _dat['buffering_factor'],
        (_eval_pts),
        sd=3
    )
    (
        lp.ggplot(
            _dat,
            lp.aes("-log10_anova_binom_p", "buffering_factor")
        )
        + lp.geom_pointdensity(
            tooltips=lp.layer_tooltips(
                ["gene_id", "gene_name", "gene_biotype"]
            ),
            show_legend=False,
        )
        + lp.geom_line(
            lp.aes("x", "y"),
            data = pl.DataFrame({"x":_eval_pts, "y": _mm}),
            color="red",
        )
        + lp.ggmarginal(sides="tr", layer=lp.geom_density())
        + lp.scale_color_viridis(option="magma")
    )
    return (moving_median,)


@app.cell(hide_code=True)
def _(buffering, good_genes5, lp, mo, np, pl):
    _data = buffering.filter(
        pl.col("gene_id").is_in(good_genes5),
        pl.col("anova_binom_p") < 1e-25, # highly significant
    ).with_columns(
        pl.col("buffering_factor").cut(np.linspace(-0.5,1.5,31), include_breaks=True),
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


@app.cell(hide_code=True)
def _(buffering, good_genes5, lp, mo, np, pl):
    _data1 = buffering.filter(
        pl.col("gene_id").is_in(good_genes5),
        pl.col("anova_binom_p") < 1e-25, # highly significant
    )
    _data2 = _data1.with_columns(
        pl.col("binom_p_gof").cut(np.geomspace(1e-10,1,31), include_breaks=True)
    ).with_columns(
        binom_p_gof = pl.col("binom_p_gof").struct.field("breakpoint"),
    ).group_by(
        "binom_p_gof"
    ).agg(
        num_genes = pl.len()
    )
    mo.vstack([
        "Checking the goodness of fit tests for the ASE (binomial) models: testing whether haplotypes are consistent in ASE.",
        lp.gggrid([
            lp.ggplot(
                _data2,
                lp.aes("binom_p_gof", "num_genes"),
            ) + lp.geom_bar(stat="identity")
            + lp.scale_x_log10(),
            lp.ggplot(
                _data1.sort("binom_p_gof").with_columns(
                    binom_p_gof = -pl.col('binom_p_gof').log10(),
                    theoretical_p = -((pl.row_index()+0.5) / pl.len()).log10(),
                ),
                lp.aes(x="theoretical_p", y="binom_p_gof"),
            )
            + lp.geom_point()
            + lp.geom_abline(intercept=0, slope=1, color='red')
            + lp.labs(x="-log10(theoretical p)", y="-log10(GOF p)")
        ]) + lp.ggsize(900, 400)
    ])
    return


@app.cell(hide_code=True)
def _(buffering, good_genes5, lp, mo, np, pl):
    _data1 = buffering.filter(
        pl.col("gene_id").is_in(good_genes5),
        pl.col("anova_binom_p") < 1e-25, # highly significant
    )
    _data2 = _data1.with_columns(
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
        lp.gggrid([
            lp.ggplot(
                _data2,
                lp.aes("deming_p_gof", "num_genes"),
            ) + lp.geom_bar(stat="identity")
            + lp.scale_x_log10(),
            lp.ggplot(
                _data1.sort("deming_p_gof").with_columns(
                    deming_p_gof = -pl.col('deming_p_gof').log10(),
                    theoretical_p = -((pl.row_index()+0.5) / pl.len()).log10(),
                ),
                lp.aes(x="theoretical_p", y="deming_p_gof"),
            )
            + lp.geom_point()
            + lp.geom_abline(intercept=0, slope=1, color='red')
            + lp.labs(x="-log10(theoretical p)", y="-log10(GOF p)")
        ]) + lp.ggsize(900, 400)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bias model results
    """)
    return


@app.cell
def _(HAPLOTYPES, json, np):
    def _():
        error_genes = []
        out = []
        eta_out = []
        beta_covs_out = {}
        with open("results/Adipose/buffering_with_bias.ndjson") as f:
            def get(field_str):
                entries = field_str.split(".")
                temp = record
                for entry in entries:
                    temp = temp[entry]
                return temp
            def sd(mat):
                return np.sqrt(np.diag(np.array(mat))).tolist()
            #for line, i  in zip(f, range(500)):
            for line in f:
                record = json.loads(line.strip())
                if record.get('error', 'error') != 'none':
                    error_genes.append(record)
                    continue

                gene_id = get('gene_id')
                out.append({
                    "gene_id": gene_id,
                    'ase_model': {
                        'beta': get("ase_model.beta"),
                        'gamma': get("ase_model.gamma"),
                        'delta': get("ase_model.delta"),
                        'beta_se': sd(get("ase_model.cov.cov_beta")),
                        'gamma_se': sd(get("ase_model.cov.cov_gamma")),
                        'delta_se': sd(get("ase_model.cov.cov_delta")),
                        "sigma_u": get("ase_model.stage2_info.sigma_u"),
                        "sigma_gamma": get("ase_model.stage2_info.sigma_gamma"),
                        "pymc_info": get("ase_model.pymc_info"),
                        "stage2_converged": get("ase_model.stage2_info.converged"),
                    },
                    "total_model": {
                        "beta": [get(f"total_model.beta.beta[{hap}].mean") for hap in HAPLOTYPES],
                        "beta_se": [get(f"total_model.beta.beta[{hap}].sd") for hap in HAPLOTYPES],
                        "diagnostic": get("total_model.diagnostic"),
                        "info": get("total_model.info"),
                    },
                    'buffering':  get("buffering"),
                    'runtime': get("runtime"),
                })

                eta_out.append({
                    "gene_id": gene_id,
                    "eta": [x['mean'] for x in get("ase_model.eta").values()],
                    "eta_se": [x['sd'] for x in get("ase_model.eta").values()],
                    "mouse_id": [x.removeprefix("eta[").removesuffix("]") for x in get("ase_model.eta").keys()]
                })
                beta_covs_out[gene_id] = get("ase_model.cov.cov_beta")
        print(f"Skipped {len(error_genes)} genes with errors")
        return out, eta_out, beta_covs_out
    buffering_with_bias_objs, eta_objs, beta_covs = _()
    return buffering_with_bias_objs, eta_objs


@app.cell
def _(buffering_with_bias_objs, pl):
    buffering_all2 = pl.DataFrame(buffering_with_bias_objs)
    return (buffering_all2,)


@app.cell
def _(eta_objs, pl):
    etas = pl.DataFrame(eta_objs).explode(['eta', 'eta_se', 'mouse_id'])
    return (etas,)


@app.cell
def _(pl):
    def extract_field(field_str):
        entries = field_str.split(".")
        temp = pl.col(entries[0])
        for entry in entries[1:]:
            temp = temp.struct.field(entry)
        return temp

    return (extract_field,)


@app.cell
def _(HAPLOTYPES, buffering_all2, extract_field, gene_annot, pl):
    buffering2 = (
        buffering_all2
        .select(
            "gene_id",
            buffering_factor = pl.col("buffering").struct.field("lambda"),
            buffering_factor_p = pl.col("buffering").struct.field("chi2_p"),
            buffering_factor_chi2 = pl.col("buffering").struct.field("chi2"),
            buffering_factor_ci_lo = pl.col("buffering").struct.field("lo"),
            buffering_factor_ci_hi = pl.col("buffering").struct.field("hi"),
            deming_Q = pl.col("buffering").struct.field("Q"),
            deming_p_gof = pl.col("buffering").struct.field("p_gof"),
            **{f"effect_{hap}": extract_field(f"ase_model.beta").list.get(i)
               for i, hap in enumerate(HAPLOTYPES)},
            **{f"effect_{hap}_se": extract_field(f"ase_model.beta_se").list.get(i)
               for i, hap in enumerate(HAPLOTYPES)},
            **{f"total_{hap}": extract_field(f"total_model.beta").list.get(i)
               for i, hap in enumerate(HAPLOTYPES)},
            **{f"total_{hap}_se": extract_field(f"total_model.beta_se").list.get(i)
               for i, hap in enumerate(HAPLOTYPES)},
            sigma_u = extract_field("ase_model.sigma_u"),
            sigma_gamma = extract_field("ase_model.sigma_gamma"),
            ase_median_rhat_eta = extract_field("ase_model.pymc_info.median_rhat_eta"),
            ase_min_ess_bulk_eta = extract_field("ase_model.pymc_info.min_ess_bulk_eta"),
            ase_min_ess_tail_eta = extract_field("ase_model.pymc_info.min_ess_tail_eta"),
            ase_n_diverging_chains = extract_field("ase_model.pymc_info.n_diverging_chains"),
            ase_stage2_converged = extract_field("ase_model.stage2_converged"),
            total_n_diverging = extract_field("total_model.info.n_diverging"),
            total_max_rhat_beta = extract_field("total_model.diagnostic.max_rhat_beta"),
            total_max_rhat_core = extract_field("total_model.diagnostic.max_rhat_core"),
            total_max_rhat_rest = extract_field("total_model.diagnostic.max_rhat_rest"),
            total_min_ess_bulk_beta = extract_field("total_model.diagnostic.min_ess_bulk_beta"),
            total_min_ess_bulk_core = extract_field("total_model.diagnostic.min_ess_bulk_core"),
            total_min_ess_bulk_rest = extract_field("total_model.diagnostic.min_ess_bulk_rest"),
            total_min_ess_tail_beta = extract_field("total_model.diagnostic.min_ess_tail_beta"),
            total_min_ess_tail_core = extract_field("total_model.diagnostic.min_ess_tail_core"),
            total_min_ess_tail_rest = extract_field("total_model.diagnostic.min_ess_tail_rest"),
        ).with_columns(
            ase_success = (
                (pl.col("ase_n_diverging_chains") == 0)
                & (pl.col("ase_median_rhat_eta") < 1.01)
                & (pl.col("ase_min_ess_bulk_eta") > 400)
                & (pl.col("ase_min_ess_tail_eta") > 400)
                & (pl.col('ase_stage2_converged'))
            ),
            total_success = (
                (pl.col("total_n_diverging") == 0)
                & (pl.col("total_max_rhat_beta") < 1.01)
                & (pl.col("total_max_rhat_core") < 1.05)
                & (pl.col("total_min_ess_bulk_beta") > 400)
                & (pl.col("total_min_ess_tail_beta") > 400)
            ),
        ).join(
            gene_annot.select("gene_id", "gene_name", "gene_biotype", "chrom"),
            "gene_id"
        )
    )
    buffering2
    return (buffering2,)


@app.cell
def _(buffering2, mo):
    _ase_success = buffering2['ase_success'].sum()
    _ase_success_pct = buffering2['ase_success'].mean()
    _total_success = buffering2['total_success'].sum()
    _total_success_pct = buffering2['total_success'].mean()
    _both_success = (buffering2['ase_success'] & buffering2['total_success']).sum()
    _both_success_pct = (buffering2['ase_success'] & buffering2['total_success']).mean()
    mo.vstack([
       f"Num ase_model success: {_ase_success} ({_ase_success_pct:0.2%})",
       f"Num total_model success: {_total_success} ({_total_success_pct:0.2%})",
       f"Num both success: {_both_success} ({_both_success_pct:0.2%})",
    ])
    return


@app.cell
def _(HAPLOTYPES, buffering2, good_genes5, lp, moving_median, np, pl):
    _dat = buffering2.filter(
        pl.col('gene_id').is_in(good_genes5),
        pl.col("ase_success"),
        pl.col("total_success"),
        pl.col("sigma_u") < 0.25,
        pl.col("sigma_gamma") < 0.25,
        #pl.col("deming_p_gof") > 0.0001,
    ).with_columns(
        log_ase_zscore = pl.max_horizontal([
            (pl.col(f'effect_{h}') / pl.col(f'effect_{h}_se')).log()
            for h in HAPLOTYPES
        ])
    ).drop_nans("log_ase_zscore")

    _eval_pts = np.linspace(_dat['log_ase_zscore'].min(), _dat['log_ase_zscore'].max(), 100)
    _mm = moving_median(
        _dat['log_ase_zscore'],
        _dat['buffering_factor'],
        (_eval_pts),
        sd=3
    )
    print(_mm)
    (
        lp.ggplot(
            _dat,
            lp.aes("log_ase_zscore", "buffering_factor")
        )
        + lp.geom_pointdensity(
            tooltips=lp.layer_tooltips(
                ["gene_id", "gene_name", "gene_biotype", "buffering_factor_p"]
            ),
            show_legend=False,
        )
        + lp.geom_line(
            lp.aes("x", "y"),
            data = pl.DataFrame({"x":_eval_pts, "y": _mm}),
            color="red",
        )
        + lp.ggmarginal(sides="tr", layer=lp.geom_density())
        + lp.scale_color_viridis(option="magma")
    )
    return


@app.cell
def _(buffering2, gene_annot, good_genes5, lp, mo, pl):
    _data1 = buffering2.filter(
        pl.col("gene_id").is_in(good_genes5),
        pl.col("ase_success"),
        pl.col("total_success"),
        pl.col("sigma_u") < 0.25,
        pl.col("sigma_gamma") < 0.25,
    ).join(gene_annot.select('gene_id', 'gene_name'), 'gene_id')
    mo.vstack([
        lp.gggrid([
            lp.ggplot(
                _data1.sort("buffering_factor_p").with_columns(
                    buffering_factor_p = -pl.col('buffering_factor_p').log10(),
                    theoretical_p = -((pl.row_index()+0.5) / pl.len()).log10(),
                ),
                lp.aes(x="theoretical_p", y="buffering_factor_p"),
            )
            + lp.geom_point(tooltips=lp.layer_tooltips(["gene_id", 'gene_name']))
            + lp.geom_abline(intercept=0, slope=1, color='red')
            + lp.labs(x="-log10(theoretical p)", y="-log10(chi2 p)")
            + lp.ggtitle("QQ plot of p-value for non-1 buffering factor.")
        ]) + lp.ggsize(500, 400)
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
    OUTLIER_MOUSE_IDS,
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
            ).filter(
                ~pl.col("mouse_id").is_in(OUTLIER_MOUSE_IDS)
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
        _hap_str = lp.as_discrete("hap_str", levels=["A", "AA", "B", "BB", "C", "CC", "D", "DD", "E", "EE", "F", "FF", "G", "GG", "H", "HH"], order=1)
        median_expr = au["norm_expr"].median()
        plt_totals = (
            lp.ggplot(
                by_hap_counts, lp.aes(x=_hap_str, y="norm_expr", color=_hap)
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
    genotypes,
    lp,
    pl,
    size_factors,
):
    def _():
        au = allele_unique.filter(gene_id=gene_selector.value).join(
            size_factors, "mouse_id"
        ).join(
            genotypes.filter(gene_id=gene_selector.value),
            "mouse_id",
        ).filter(
            ~pl.col("mouse_id").is_in(OUTLIER_MOUSE_IDS),
            pl.col("genotype_confidence") > 0.9,
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
                        index=["mouse_id", "total", "unspecific", "incompatible", "genotype_confidence"],
                        variable_name="class",
                        value_name="expr",
                    )
                    .sort("mouse_id")
                    .with_columns(
                        mouse_idx = pl.col("mouse_id").is_first_distinct().cast(int).cum_sum(),
                        total_expr = pl.col("expr").sum().over("mouse_id"),
                    )
                )
                plot_data.append(hap_data.with_columns(hap1=pl.lit(hap1), hap2=pl.lit(hap2)))

        _class = lp.as_discrete(
            "class", levels=HAPLOTYPES + ["unspecific", "incompatible"], order=1
        )
        df = pl.concat(plot_data)
        summary_df = (
            df
            .group_by(["mouse_id", "mouse_idx", "genotype_confidence", "hap1", "hap2"])
            .agg(
                total_expr = pl.col("expr").sum(),
                hap1_expr = pl.col("expr").filter(pl.col("class") == pl.col("hap1")).first(),
                hap2_expr = pl.col("expr").filter(pl.col("class") == pl.col("hap2")).first(),
                unspecific_expr = pl.col("expr").filter(pl.col("class") == "unspecific").first(),
                incompatible_expr = pl.col("expr").filter(pl.col("class") == "incompatible").first(),
            )
        )
        plt = (
            lp.ggplot(
                df,
                lp.aes(
                    x="mouse_idx", y="expr", color=_class, fill=_class
                ),
            )
            + lp.facet_grid(x="hap1", y="hap2")
            + lp.geom_bar(
                stat="identity",
                tooltips=lp.layer_tooltips(),
            )
            + lp.geom_bar(
                lp.aes("mouse_idx", y="total_expr"), # dummy bar for tooltips
                data = summary_df,
                inherit_aes=False,
                tooltips=lp.layer_tooltips(["mouse_id", "genotype_confidence"])
                    .line("@hap1: @hap1_expr")
                    .line("@hap2: @hap2_expr")
                    .line("unspecific: @unspecific_expr")
                    .line("incompat: @incompatible_expr")
                ,
                stat = 'identity',
                color="transparent",
                fill="transparent",
                alpha=0.0,
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


@app.cell(hide_code=True)
def _(buffering2, gene_selector, mo):
    _dat = buffering2.filter(gene_id = gene_selector.value).to_dicts()[0]
    mo.vstack([
        f"gene_id: {_dat['gene_id']} | {_dat['gene_name']}",
        f"sigma_u = {_dat['sigma_u']:0.2f} | sigma_gamma: {_dat['sigma_gamma']:0.2f}",
        f"buffering factor: {_dat['buffering_factor']:0.2f} ({_dat['buffering_factor_ci_lo']:0.2f}-{_dat['buffering_factor_ci_hi']:0.2f})"
    ])
    return


@app.cell(hide_code=True)
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
            + lp.ggtb()
            + lp.ggtitle("Estimated effects by model type")
        )
    mo.vstack([
        "Plot the ASE (binomial) model fit parameters versus the total counts (NB GLM) fit parameters. In dash red, the buffering fit line and in black the reference diagonal line. Note that the buffering fit line takes into account covariance of the estimate, which is typically substantial and means that the fit may not line between the points (it's intercept is always 0).",
        _(),
    ])
    return


@app.cell
def _(HAPLOTYPES, HAPLOTYPE_COLORS, buffering2, gene_selector, lp, mo, pl):
    def _():
        data = buffering2.filter(gene_id = gene_selector.value)
        buff_factor = data['buffering_factor'][0]
        df = pl.DataFrame(dict(
            hap = [hap for hap in HAPLOTYPES],
            x = data[[f'effect_{hap}' for hap in HAPLOTYPES]].to_numpy()[0], # H is refernece, always 0
            x_se = data[[f'effect_{hap}_se' for hap in HAPLOTYPES]].to_numpy()[0],
            y = data[[f'total_{hap}' for hap in HAPLOTYPES]].to_numpy()[0],
            y_se = data[[f'total_{hap}_se' for hap in HAPLOTYPES]].to_numpy()[0],
        )).with_columns(
            x_min = pl.col("x") - 1.96*pl.col("x_se"),
            x_max = pl.col("x") + 1.96*pl.col("x_se"),
            y_min = pl.col("y") - 1.96*pl.col("y_se"),
            y_max = pl.col("y") + 1.96*pl.col("y_se"),
        )
        xmin, xmax = min(df['x_min']), max(df['x_max'])
        ymin, ymax = min(df['y_min']), max(df['y_max'])
        lims = min(ymin,xmin), max(xmax, ymax)
        fit_lines = pl.DataFrame(dict(
            slope = [1.0, buff_factor],
            intercept = [0, 0],
            fit_type = ["no buffering", "fit w/ buffering"],
        ))
        return (
            lp.ggplot(df, lp.aes("x", "y", color="hap"))
            + lp.geom_point(
                tooltips=lp.layer_tooltips(["hap"])
            )
            + lp.geom_errorbar(lp.aes(xmin="x_min", xmax="x_max"))
            + lp.geom_errorbar(lp.aes(ymin="y_min", ymax="y_max"))
            + lp.geom_point(data={"x": [0], "y": [0]}, mapping=lp.aes("x", "y"), alpha = 0, color="black") #ensure 0,0 is in view
            + lp.labs(x="beta (ASE)", y="beta (total)")
            + lp.scale_color_manual(
                values=HAPLOTYPE_COLORS,
                breaks=HAPLOTYPES,
            )
            + lp.geom_abline(lp.aes(linetype="fit_type", slope="slope", intercept="intercept"), inherit_aes=False, data=fit_lines)
            #+ lp.geom_abline(lp.aes(linetype="fit"), slope=buff_factor, intercept=0)
            #+ lp.geom_abline(lp.aes(linetype="diagonal"), slope=1, intercept=0)
            + lp.scale_linetype_manual(name="fit type", values={"fit w/ buffering": 2, "no buffering": 3})
            + lp.coord_fixed()
            + lp.xlim(*lims)
            + lp.ylim(*lims)
            + lp.ggtb()
            + lp.ggtitle("Estimated effects by model type")
        )
    mo.vstack([
        "BUFFERING BIAS MODEL: Plot the ASE model fit parameters versus the total counts fit parameters.",
        _(),
    ])
    return


@app.cell
def _(HAPLOTYPES, buffering2, etas, gene_selector, genotypes, lp, pl):
    # Plot the etas for this gene by genotype
    def _():
        max_mice_per_group=8
        gene_betas = (
            buffering2
            .filter(gene_id = gene_selector.value)
            .unpivot(
                on=[f"effect_{h}" for h in HAPLOTYPES],
                variable_name="hap",
                value_name="beta",
            )
            .with_columns(
                pl.col("hap").str.strip_prefix("effect_")
            )
        )
        gene_etas = (
            etas
                .filter(gene_id = gene_selector.value)
                .join(genotypes.filter(gene_id = gene_selector.value), "mouse_id")
                .with_columns(
                    eta_lo = pl.col("eta") - 1.96 * pl.col("eta_se"),
                    eta_hi = pl.col("eta") + 1.96 * pl.col("eta_se"),
                    mouse_num = pl.row_index().over("genotype").mod(max_mice_per_group).cast(int),
                ).with_columns(
                    mouse_num = (pl.col("mouse_num") - pl.col("mouse_num").max().over("genotype")//2),
                    hap1 = pl.col("genotype").str.slice(0,1),
                    hap2 = pl.col("genotype").str.slice(1,1),
                )
                .sort("genotype")
                .join(gene_betas.select(hap1="hap", beta1="beta"), on="hap1")
                .join(gene_betas.select(hap2="hap", beta2="beta"), on="hap2")
                .with_columns(
                    beta_fit = pl.col("beta1") - pl.col("beta2"),
                )
        )
        plt = (
            lp.ggplot(
                gene_etas,
                lp.aes("genotype", "eta", group="mouse_num")
            )
            + lp.geom_errorbar(lp.aes(ymin="eta_lo", ymax="eta_hi"), position=lp.position_dodge(width=0.7), alpha=0.4)
            + lp.geom_point(position=lp.position_dodge(width=0.7), tooltips=lp.layer_tooltips(["mouse_id", "genotype"]))
            + lp.geom_crossbar(lp.aes(y="beta_fit"), color="red")
            + lp.labs(y="eta")
            + lp.ggsize(800, 400)
        )
        #return gene_etas
        return plt
    _()
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
    sim_source_counts = pl.read_csv("processed/simulated_reads/source_counts_by_gene.txt", separator="\t")
    return (sim_source_counts,)


@app.cell
def _(good_genes4, mo):
    mo.vstack([f"Final total of genes selected for use: {len(good_genes4)}"])
    return


if __name__ == "__main__":
    app.run()
