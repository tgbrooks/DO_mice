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

    import yaml

    from util.compressed_emase import load_compressed_emase
    from util.compat_classes import get_gene_class_counts, get_gene_totals, sparse_any
    from ase_buffering.ase_model import (
        make_ase_model,
        summarize_read_classes,
        summarize_bias_rates,
        estimate_class_proportions,
        estimate_class_alpha,
    )
    from ase_buffering.total_counts_model import make_total_model


@app.cell
def _():
    config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
    HAPLOTYPES = config["haplotypes"].split(",")
    HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}
    SEX_TO_NUM = {"F": 0, "M": 1}
    return HAPLOTYPES, HAP_TO_HAPNUM, config


@app.cell
def _():
    import marimo as mo
    import lets_plot as lp

    return lp, mo


@app.cell
def _():
    size_factors = pl.read_csv("processed/Adipose/size_factors.txt", separator="\t")
    return (size_factors,)


@app.cell
def _(config):
    OUTLIER_MOUSE_IDS = config["outlier_ids"]
    return (OUTLIER_MOUSE_IDS,)


@app.cell
def _(HAPLOTYPES, lp):
    HAPLOTYPE_COLORS = lp.scale_color_brewer(palette="Dark2").palette(len(HAPLOTYPES))
    return (HAPLOTYPE_COLORS,)


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
    return (gene_annot,)


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
    allele_unique = pl.read_parquet("processed/Adipose/allele_unique_reads.parquet")
    allele_unique
    return (allele_unique,)


@app.cell
def _():
    effective_lengths = pl.read_parquet(
        f"processed/Adipose/effective_lengths.by_gene.parquet"
    )
    return (effective_lengths,)


@app.cell
def _():
    genotypes = pl.read_parquet("processed/genotypes.parquet")
    return (genotypes,)


@app.cell
def _(allele_unique):
    mouse_ids = sorted(allele_unique["mouse_id"].unique())
    return (mouse_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model ASE with bias correction
    ASE is fundamentally estimated with the ratio of reads that align uniquely to one haplotype versus the other. However, some haplotypes may be easier to distinguish than others: the main culprit being that longer haplotypes naturally produce more reads that are unique, creating a bias in the ASE estimation. A further complication is that RNA-seq reads are not uniformly distributed across the transcript. The combination of these makes it difficult to analytically compute the bias. Instead, we want to estimate it from the data, which is made tricky by the fact that the data also has allele-specific expression which looks very similar.

    ## Modelling from homozygotes

    The key insight is that if a sample has diplotype AA at a gene, then we can measure what fraction of its reads would have been informative had it had an AB diplotype. Call that $f_{AB}$: the fraction of A haplotype reads that are distinguishable from B. This is estimatable just from alignment statistics. In comparison, $f_{BA}$ would be the fraction of reads of a B haplotype that are distinguishable from A. $f_{BA}$ can likewise be estimated from the BB haplotypes. These numbers are not necessarily equal and their ratio $f_{AB} / f_{BA}$ is the bias in allele unique read count ratios in the heterozygous AB samples.

    While this works, we have 8 haplotypes and so only 1/64th of our data is AA: many genes have just 0, 1, or 2 samples with in at least one homozygous. That estimate is therefore too noisy.

    ## Modelling with all diplotypes simultaneously

    We can extend this idea to use all the diplotypes simultaneously.
    Consider now a sample of AC genotype.
    The difficulty we didn't face with homozygotes is that we don't know whether a given read came from A or from C.
    But we can still use it to inform estimates of $f_{AB}$ and $f_{CB}$, if we simultaneously estimate the proportion of reads coming from $A$ and $C$.
    In fact, that proportion is the actual value we are interested in, call it $p_{AC}$: the expected fraction of reads from AC diplotypes that came from $A$. Likewise, $p_{CA} = 1 - p_{AC}$ is the fraction of reads from CA (which is the same as AC) that came from $C$.
    So $p_{AB}$, $p_{AC}$, etc. are the values we are truly interested in and $f_{AB}$, etc. are nuisance parameters.

    ## The model

    Let $r_{i, h}$ be the fraction of reads from sample $i$ that are dinstinguishably *not* from haplotype $h$.
    This is an observed variable.
    Also observed are the diplotypes, $g m$: the two haplotypes of sample $i$.
    Then

    $$
    \begin{align}
    \mbox{E}[r_{i,h}]
        &= f_{g h} p_{g m} + f_{m h} p_{m g} \\
        &= f_{g h} p_{g m} + f_{m h} (1- p_{g m}).
    \end{align}
    $$
    For homozygotes $g g$ this simplifies to:
    $$
    \mbox{E}[r_{i,h}] = f_{g h}.
    $$

    There are then $(8 choose 2)=28$ parameters $p_{g m}$ and $64$ nuisance parameters $f_{g m}$ to fit.

    As in the unbiased model, we further simplify by assuming
    $$
    p_{h_i h_j} = \mbox{expit}(\beta_i - \beta_j)
    $$
    giving just 8 parameters $\beta_i$ (of which, one is redundant and can be assumed to be zero), as well as the 64 nuisance parameters.
    Moreover, we have now $8$ observations per sample: $r_{i,h}$ for all possible values of $h$.
    Notably, some of those are trivial: $r_{i, A} = 0$ if the sample is $AA$ by construction (though not actually so in real data due to either sequencing errors that happen to match other haplotypes or misgenotyping).
    Furthermore, they're very much not independent.

    ## Multinomial formulation

    Instead, we can address multiple problems by instead considering a multinomial model over the haplotype-compatibility classes of the reads.
    A read can be alignment compatible with some subset of the 8 haplotypes and we consider all reads that are compatible with the same set of haplotypes to be in the same haplotype-compatibility class $S \subset \{A, \ldots, H\}$.
    Then each sample (and for each gene), we observe the number of reads that are in class $S$.
    We'll number the distinct classes $s = 1, \ldots, K$.
    For computational efficiency, we restrict $K$ to be just the compatibility classes expressed in any sample ($K \leq 2^8$) which is typically between 15-35, varying by gene.
    So our data is a vector $\bm{y}_{i}$ for sample $i$ of read counts for each class $s$ and let $Y_i$ be the total read counts for sample $i$, so $Y_i = \sum_s y_{is}$.

    Now we model for sample $i$ with genotype $g h$.
    $$
    \begin{align}
     \bm{y}_i | Y_i &\sim \mbox{Multinomial}(Y_i, \mbox{c}_i) \\
     c_{is} &:= q_{gs} p_i + q_{hs} (1 - p_i) \\
     p_{is} &:= \mbox{expit}(\beta_g - \beta_h).
    \end{align}
    $$
    The parameters to be fit are $\beta_g$ (haplotype-specific expression propensity) and $q_{gs}$, the proportion of reads from haplotype $g$ that map to class $s$.
    The eight $\beta_g$ parameters are the effects we care about and $8K$ $q_{gs}$ values are nuisance parameters.

    Notably, the model does not use the mapping of classes to haplotype sets!
    Indeed, being ignorant of this allows it to infer actual haplotype-specific mapping patterns including ones caused by  reference genomes errors or sequencing errors.
    """)
    return


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
def _(allele_unique, args, gene_annot, mo):
    _vals = (
        allele_unique.select("gene_id")
        .unique()
        .join(gene_annot.select("gene_id", "gene_name"), "gene_id")
        .sort("gene_id")
        .drop_nulls()
    )
    _names = list(
        _vals.select(name=pl.col("gene_id") + " | " + pl.col("gene_name")).drop_nulls()[
            "name"
        ]
    )
    gene_ids = list(_vals["gene_id"])
    _options = {_name: _id for _name, _id in zip(_names, gene_ids)}
    len(gene_ids)
    if args.gene_id:
        assert (
            args.gene_id in gene_ids
        ), f"Provided gene id must be in gene ids (e.g. {gene_ids[:10]})"
        (default,) = [n for n, id in _options.items() if id == args.gene_id]
    else:
        default = None
    gene_selector = mo.ui.dropdown(_options, value=default, searchable=True)
    mo.sidebar(
        [
            "Gene selector:",
            gene_selector,
        ]
    )
    return (gene_selector,)


@app.function
def threshold_class_counts(class_counts, min_tot_reads):
    import util.compat_classes

    counts = class_counts.counts.sum(axis=0)
    selected = counts >= min_tot_reads
    return util.compat_classes.CompatClassesDf(
        compat=class_counts.compat[selected],
        counts=class_counts.counts[:, selected],
        ids=class_counts.ids,
        haplotypes=class_counts.haplotypes,
    )


@app.cell
def _(all_compatibility_classes, effective_lengths, gene_selector, tx_annot):
    gene_class_counts_raw = get_gene_class_counts(
        gene_selector.value, tx_annot, all_compatibility_classes
    )
    gene_class_counts = gene_class_counts_raw
    # gene_class_counts = threshold_class_counts(
    #    gene_class_counts_raw,
    #    min_tot_reads = 500,
    # )
    gene_effective_lengths = effective_lengths.filter(
        gene_id=gene_selector.value
    ).select("mouse_id", "haplotype", "effective_length")
    return gene_class_counts, gene_effective_lengths


@app.cell
def _(gene_class_counts, gene_effective_lengths, gene_selector, genotypes):
    ase_model = make_ase_model(
        gene_selector.value,
        gene_class_counts,
        genotypes.select(
            "gene_id", "mouse_id", "genotype_confidence", diplotype="genotype"
        ),
        gene_effective_lengths,
        config=dict(
            sigma_u_sd=0.5,
            q_mode="adaptive_centered",
            sigma_q_sd=0.15,
            beta_sd=0.5,
            gamma_mode="zero",
            sigma_gamma_sd=1.0,
            min_genotype_confidence=0.9,
        ),
    )
    return (ase_model,)


@app.cell
def _(HAPLOTYPES, HAP_TO_HAPNUM, gene_class_counts, gene_selector, genotypes):
    # For plotting
    n_samples = int(gene_class_counts.counts.shape[0])
    n_classes = int(gene_class_counts.compat.shape[0])
    n_haps = len(HAPLOTYPES)
    gene_diplotypes = (
        genotypes.filter(gene_id=gene_selector.value)
        .select(
            "mouse_id", pl.col("genotype").alias("diplotype"), "genotype_confidence"
        )
        .join(
            pl.DataFrame({"mouse_id": gene_class_counts.ids}),
            "mouse_id",
            maintain_order="right",
        )
        .with_columns(
            hap1=pl.col("diplotype").str.slice(0, 1).replace_strict(HAP_TO_HAPNUM),
            hap2=pl.col("diplotype").str.slice(1, 1).replace_strict(HAP_TO_HAPNUM),
        )
    )
    hap1 = gene_diplotypes["hap1"].to_numpy()
    hap2 = gene_diplotypes["hap2"].to_numpy()
    return gene_diplotypes, hap1, hap2, n_classes, n_haps


@app.cell
def _(ase_model, mo):
    mo.mermaid(pm.model_to_mermaid(ase_model))
    return


@app.cell
def _(ase_model):
    # Fit the model
    _start = time.time()
    with ase_model:
        idata_ase = pm.sample(
            300,
            chains=6,
            cores=6,
            blas_cores=1,
            random_seed=100,
            nuts_sampler="nutpie",
            # nuts={"adaptation": "low_rank"},
            # nuts={"adaptation": "full"},
            # nuts=dict(target_accept=0.8),
        )
    _end = time.time()
    ase_model_time = _end - _start
    return ase_model_time, idata_ase


@app.cell
def _(idata_ase):
    az.plot_trace_dist(idata_ase, var_names=["beta"], compact=False)
    return


@app.cell
def _(idata_ase):
    az.plot_forest(
        idata_ase,
        var_names=["beta"],
        combined=True,
        ci_probs=[0.5, 0.95],
        figure_kwargs={"figsize": (10, 3)},
    )
    return


@app.cell
def _(idata_ase, lp, n_haps):
    _gamma = (
        # (idata_ase["posterior"]["sigma_gamma"] * ) # using centered version now
        idata_ase["posterior"]["gamma_raw"]
        .to_numpy()
        .reshape((-1, (n_haps * (n_haps - 1) // 2)))
    )
    _names = idata_ase["posterior"]["gamma_raw"].coords["hap_pairs"].data
    _df = (
        pl.DataFrame({f"gamma[{n}]": _gamma[:, i] for i, n in enumerate(_names)})
        .unpivot(on=[f"gamma[{n}]" for n in _names])
        .group_by("variable")
        .agg(
            mean=pl.col("value").mean(),
            low=pl.col("value").quantile(0.05),
            hi=pl.col("value").quantile(0.95),
        )
        .sort("variable")
    )
    (
        lp.ggplot(_df, lp.aes(x="mean", y="variable"))
        + lp.geom_point(tooltips=lp.layer_tooltips(["variable"]))
        + lp.geom_errorbar(lp.aes(xmin="low", xmax="hi"))
        + lp.labs(x="gamma")
    )
    return


@app.cell
def _(idata_ase):
    az.summary(idata_ase, var_names="beta", filter_vars="like")
    return


@app.cell
def _(idata_ase):
    az.summary(idata_ase, var_names=["sigma_"], filter_vars="like")
    return


@app.cell(hide_code=True)
def _(HAPLOTYPES, gene_class_counts, idata_ase, lp, n_classes):
    q_hat = np.mean(idata_ase["posterior"]["q"], axis=(0, 1))
    q_hat_se = np.std(idata_ase["posterior"]["q"], axis=(0, 1))
    # q_hat = [np.mean(samples['posterior'][f'q_{i}'], axis=(0,1)) for i in range(n_haps)]
    _dat = pl.concat(
        [
            pl.DataFrame(
                {
                    "haplotype": hap,
                    "class": range(len(q_hat[i])),
                    "q_hat": q_hat[i].to_numpy(),
                    "q_hat_se": q_hat_se[i].to_numpy(),
                    "is_leak": ~gene_class_counts.compat[:, i],
                    "compat_with": [
                        "".join(
                            hap2
                            for j, hap2 in enumerate(HAPLOTYPES)
                            if gene_class_counts.compat[k, j]
                        )
                        for k in range(n_classes)
                    ],
                }
            )
            for i, hap in enumerate(HAPLOTYPES)
        ]
    ).with_columns(
        y=pl.col("q_hat"),
        ymin=pl.col("q_hat") - 1.96 * pl.col("q_hat_se"),
        ymax=pl.col("q_hat") + 1.96 * pl.col("q_hat_se"),
    )
    (
        lp.ggplot(_dat, lp.aes(x="class", y="y", color="is_leak"))
        + lp.facet_wrap("haplotype")
        + lp.geom_point(
            tooltips=lp.layer_tooltips(
                ["haplotype", "class", "q_hat", "ymin", "ymax", "compat_with"]
            )
        )
        + lp.geom_errorbar(lp.aes(ymin="ymin", ymax="ymax"))
        + lp.scale_color_manual(breaks=[False, True], values=["black", "red"])
        + lp.ggtb()
        + lp.labs(y="q_hat")
        + lp.scale_y_log10()
        + lp.ggtitle("Class proportions by haplotype")
    )
    return


@app.cell
def _(gene_class_counts, idata_ase, lp):
    class_summaries = summarize_read_classes(idata_ase, gene_class_counts)
    max_leak = class_summaries["leak_to"].max()
    lp.gggrid(
        [
            (
                lp.ggplot(
                    class_summaries,
                    lp.aes(y="source_hap", x="out_hap", fill="out_class"),
                )
                + lp.geom_tile()
                + lp.scale_fill_viridis(option="magma", limits=[0, 1])
                + lp.ggtitle("Read compatibility matrix")
            ),
            (
                lp.ggplot(
                    class_summaries,
                    lp.aes(x="out_hap", y="source_hap", fill="leak_to"),
                )
                + lp.geom_tile()
                + lp.scale_fill_viridis(option="magma")
                + lp.ggtitle(f"Leaks - max leak = {max_leak:0.2%}")
            ),
        ]
    ) + lp.ggsize(900, 500)
    return


@app.cell(hide_code=True)
def _(HAPLOTYPES, gene_class_counts, hap1, hap2, idata_ase, lp, mo):
    # For each diplotype, plot the fit percentage of reads that are allele-unique for both haplotypes
    def _():
        temp = []
        p = idata_ase["posterior"]["p"].values
        q = idata_ase["posterior"]["q"].values
        class_props = (
            p[:, :, :, None] * q[:, :, hap1] + (1 - p)[:, :, :, None] * q[:, :, hap2]
        )
        for i, h1 in enumerate(HAPLOTYPES):
            for j, h2 in enumerate(HAPLOTYPES):
                if h1 == h2:
                    continue
                # NOTE: these predictions include 'leaks'
                h1_unique_classes = gene_class_counts.compat[:, i] & (
                    ~gene_class_counts.compat[:, j]
                )
                h2_unique_classes = gene_class_counts.compat[:, j] & (
                    ~gene_class_counts.compat[:, i]
                )
                selected1 = (hap1 == i) & (hap2 == j)
                selected2 = (hap1 == j) & (hap2 == i)
                selected = selected1 | selected2

                # Predicted uniquely mapping counts for each allele
                h1_uniques = (
                    class_props[:, :, selected][..., h1_unique_classes]
                    .sum(axis=3)
                    .mean(axis=(0, 1))
                )
                h2_uniques = (
                    class_props[:, :, selected][..., h2_unique_classes]
                    .sum(axis=3)
                    .mean(axis=(0, 1))
                )

                tots = gene_class_counts.counts[selected].sum(axis=1)
                actual_h1_unique_counts = (
                    gene_class_counts.counts[selected][:, h1_unique_classes].sum(axis=1)
                    / tots
                )
                actual_h2_unique_counts = (
                    gene_class_counts.counts[selected][:, h2_unique_classes].sum(axis=1)
                    / tots
                )
                pred_allele_ratio = np.empty(p.shape[2])
                pred_allele_ratio[selected1] = p[:, :, selected1].mean(axis=(0, 1))
                pred_allele_ratio[selected2] = 1 - p[:, :, selected2].mean(axis=(0, 1))
                actual_allele_ratio = (actual_h1_unique_counts + 1e-9) / (
                    actual_h2_unique_counts + actual_h1_unique_counts + 2e-9
                )
                temp.append(
                    pl.DataFrame(
                        {
                            "hap1": h1,
                            "hap2": h2,
                            "mouse_ids": np.array(list(gene_class_counts.ids))[
                                selected
                            ],
                            "pred_hap1_unique_counts": h1_uniques,
                            "pred_hap2_unique_counts": h2_uniques,
                            "pred_allele_ratio": pred_allele_ratio[selected],
                            "actual_hap1_unique_counts": actual_h1_unique_counts,
                            "actual_hap2_unique_counts": actual_h2_unique_counts,
                            "actual_allele_ratio": actual_allele_ratio,
                        }
                    )
                )
        return pl.concat(temp)

    pred_allele_unique = _()
    _res = pred_allele_unique.unpivot(
        index=["hap1", "hap2"],
        on=[
            "pred_hap1_unique_counts",
            "actual_hap1_unique_counts",
            # Only show hap1 values - the lower/upper triangular show opposite haps
            # "pred_hap2_unique_counts",
            # "actual_hap2_unique_counts",
        ],
        variable_name="var",
        value_name="frac",
    ).with_columns(
        type=pl.col("var").str.split("_").list.get(0),
        hap=pl.col("var").str.split("_").list.get(1),
    )
    _res2 = pred_allele_unique.unpivot(
        index=["hap1", "hap2"],
        on=["pred_allele_ratio", "actual_allele_ratio"],
        variable_name="var",
        value_name="allele_ratio",
    ).with_columns(
        type=pl.col("var").str.split("_").list.get(0),
    )
    mo.vstack(
        [
            lp.ggplot(_res, lp.aes(x="type", y="frac", color="type"))
            + lp.geom_point(
                tooltips=lp.layer_tooltips(["hap1", "hap2", "type", "frac"]),
                position=lp.position_dodge(),
            )
            + lp.facet_grid(x="hap1", y="hap2")
            + lp.labs(x="type", y="fraction unique")
            + lp.ggsize(700, 700),
            lp.ggplot(_res2, lp.aes(x="type", y="allele_ratio", color="type"))
            + lp.geom_point(
                tooltips=lp.layer_tooltips(["hap1", "hap2", "type", "allele_ratio"]),
            )
            + lp.facet_grid(x="hap1", y="hap2")
            + lp.labs(x="type", y="allele ratio")
            + lp.ggsize(700, 700),
            pred_allele_unique,
        ]
    )
    return


@app.cell(hide_code=True)
def _(HAPLOTYPES, gene_class_counts, hap1, hap2, idata_ase, lp, n_classes):
    def _():
        p = idata_ase["posterior"]["p"].values
        q = idata_ase["posterior"]["q"].values
        class_props = (
            p[:, :, :, None] * q[:, :, hap1] + (1 - p)[:, :, :, None] * q[:, :, hap2]
        ).mean(axis=(0, 1))
        # Compare predicted class props to homozygous class props
        # Homozygous directly informs class props without any inference of proportions
        # so we expect these to be close if there are a good number of homozygous
        homo = hap1 == hap2
        temp = []
        for i in range(n_classes):
            temp.append(
                pl.DataFrame(
                    {
                        "mouse_id": np.array(list(gene_class_counts.ids))[homo],
                        "hap": np.array(HAPLOTYPES)[hap1[homo]],
                        "class": i,
                        "actual_prop": gene_class_counts.counts[homo, i]
                        / gene_class_counts.counts[homo, :].sum(axis=1),
                        "pred_prop": class_props[homo, i],
                    }
                )
            )
        df = pl.concat(temp)
        return df

    _compat_with = {
        k: "".join(
            hap2 for j, hap2 in enumerate(HAPLOTYPES) if gene_class_counts.compat[k, j]
        )
        for k in range(n_classes)
    }
    _df = (
        _()
        .group_by("hap", "class")
        .agg(
            pl.col("actual_prop").mean(),
            pl.col("pred_prop").mean(),
        )
        .with_columns(
            pl.col("class").replace_strict(_compat_with, return_dtype=str),
        )
    )
    (
        lp.ggplot(_df, lp.aes("actual_prop", "pred_prop", color="class"))
        + lp.geom_abline(intercept=1.0, slope=1, color="black", linestyle=2)
        + lp.geom_point(
            tooltips=lp.layer_tooltips(["class", "actual_prop", "pred_prop"])
        )
        + lp.scale_x_log10()
        + lp.scale_y_log10()
        + lp.facet_wrap("hap")
        + lp.labs(x="actual prop (in homozygous)", y="predicted prop")
        + lp.ggtb()
        + lp.ggtitle("Alignment classes in homozygous versus predicted")
    )
    return


@app.cell
def _(HAPLOTYPES, HAPLOTYPE_COLORS, gene_class_counts, idata_ase, lp):
    # Plot fit allele-unique bias rates
    _df = summarize_bias_rates(idata_ase, gene_class_counts)
    (
        lp.ggplot(
            _df, lp.aes(x="frac_unique", y="reverse_frac_unique", color="source_hap")
        )
        + lp.geom_point(
            tooltips=lp.layer_tooltips(
                [
                    "source_hap",
                    "other_hap",
                    "frac_unique",
                    "reverse_frac_unique",
                    "allele_unique_bias",
                ]
            )
        )
        + lp.geom_abline(intercept=0, slope=1)
        + lp.scale_color_manual(breaks=HAPLOTYPES, values=HAPLOTYPE_COLORS)
        + lp.ggtitle("Biases in rate of producing allele unique reads")
    )
    return


@app.cell
def _(idata_ase):
    az.plot_energy(idata_ase)
    return


@app.cell
def _():
    # Check for correlations across variables
    import itertools

    ase_raw_vars = ["q_logit", "delta", "eta", "sigma_gamma", "sigma_u"]

    def check_posterior_correlations(idata, raw_vars, chain):
        var_names = []
        temp = []
        for var in raw_vars:
            coord_names = list(idata["posterior"][var].coords)[2:]
            for coords in itertools.product(
                *(idata["posterior"][var].coords[n].data for n in coord_names)
            ):
                var_name = f"{var}:{''.join(f'[{c}]' for c in coords)}"
                temp.append(
                    idata["posterior"][var]
                    .sel(chain=chain, **{n: c for n, c in zip(coord_names, coords)})
                    .data.flatten()
                )
                var_names.append(var_name)
        corrs = np.corrcoef(np.array(temp))
        return var_names, corrs

    return ase_raw_vars, check_posterior_correlations


@app.cell
def _(ase_raw_vars, check_posterior_correlations, idata_ase):
    var_names, corrs = check_posterior_correlations(idata_ase, ase_raw_vars, chain=0)
    _offdiag = corrs - np.eye(corrs.shape[0])
    max_cols = np.argmax(_offdiag, axis=1)
    max_row = np.argmax(_offdiag[np.arange(len(max_cols)), max_cols])
    max_col = max_cols[max_row]
    var_names[max_row], var_names[max_col], corrs[max_row, max_col]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Total counts model
    """)
    return


@app.cell
def _(diplotypes, gene_class_counts, gene_selector):
    gene_totals = get_gene_totals(gene_selector.value, gene_class_counts, diplotypes)
    return


@app.cell
def _(size_factors):
    # phenotypes were acquired from Dryad  and extracted from Rdata format:
    # https://datadryad.org/dataset/doi:10.5061/dryad.pj105
    all_pheno = (
        pl.read_csv("phenotypes.csv.gz")
        .rename({"mouse.id": "mouse_id"})
        .with_columns(
            pl.col("DOwave").cast(str).cast(pl.Enum([str(x) for x in range(1, 6)]))
        )
        .join(size_factors, "mouse_id")
    )
    return (all_pheno,)


@app.cell
def _(all_pheno, diplotypes, gene_class_counts, gene_selector):
    total_model = make_total_model(
        gene_selector.value, gene_class_counts, diplotypes, all_pheno
    )
    return (total_model,)


@app.cell
def _(mo, total_model):
    mo.mermaid(pm.model_to_mermaid(total_model))
    return


@app.cell
def _(total_model):
    # Fit the total model
    _start = time.time()
    with total_model:
        idata_totals = pm.sample(
            300,
            random_seed=101,
            nuts_sampler="nutpie",
            # nuts={"adaptation": 'low_rank'},
        )
    _end = time.time()
    total_model_time = _end - _start
    return idata_totals, total_model_time


@app.cell
def _(idata_totals):
    az.plot_forest(
        idata_totals,
        var_names=["beta"],
        combined=True,
        ci_probs=[0.5, 0.95],
        figure_kwargs={"figsize": (10, 3)},
    )
    return


@app.cell
def _(idata_totals):
    pl.DataFrame(az.summary(idata_totals, ["beta"], filter_vars="like").reset_index())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Buffering model

    Synthesizing the ASE and total counts models gives buffering.
    Assume that both beta estimates come from some underlying true effect together with a lambda scaling ratio (the buffering).

    $$
    \bm{\hat \beta}_{ASE} = \bm{\beta} + \bm{\epsilon} \\
    \bm{\hat \beta}_{tot} = \lambda \bm{\beta}  + \bm{\epsilon'}
    $$
    where $\epsilon, \epsilon'$ are the measurement errors.
    """)
    return


@app.function
def covariance_deming_regression(b1, b2, V1, V2, grid=np.linspace(-2, 3, 2001)):
    # Function written by Claude Opus 5 to perform Deming-like regression when we have
    # two k-vectors beta_1 and beta_2 each with known covariance matrices V_1 and V_2
    # We assume no cross-covariance between beta_1 and beta_2
    # Model: beta_1 = theta + epsilon_1
    #        beta_2 = lambda * theta + epsilon_2
    #        epsilon_i ~ N(0, V_i)
    # So theta is the underlying 'true' parameter vector and lambda is the scaling between the two
    #
    # Q(lambda) := d(lambda)^T S(lambda)^-1 d(lambda)
    #   d(lambda) := beta_2 - lambda beta_1
    #   S(lambda) := V_2 + lambda^2 V_1
    # so Q(lambda) is profile deviance (theta is profiled out) given covariances.
    # Our solution is then the value of lambda that minimizes Q.
    p = len(b1)
    assert len(b1) == len(b2)
    # Cross-covariance between betas, assumed to be zero
    C = np.zeros((p, p))
    Cs = C + C.T

    def Q(lam):
        # objective function
        d = b2 - lam * b1
        S = V2 + lam**2 * V1 - lam * Cs
        e = np.linalg.eigh(S)
        k = e.eigenvalues > np.max(e.eigenvalues) * 1e-8  # generalized inverse
        return np.sum(
            d[k].T
            @ e.eigenvectors[:, k].T
            @ e.eigenvectors[:, k]
            @ d[k]
            / e.eigenvalues[k]
        )

    # First minimize Q over a discrete grid
    q = np.array([Q(lam) for lam in grid])
    i = np.argmin(q)
    # Then find the minimum by numeric search in the neighborhood of that grid point
    start = grid[max(i - 1, 0)]
    end = grid[min(i + 1, len(grid) - 1)]
    opt = scipy.optimize.minimize(Q, x0=grid[i], bounds=[(start, end)])
    # Now compute a profile confidence interval on lambda
    keep = q <= opt.fun + scipy.stats.chi2.ppf(0.95, df=1)
    r = np.linalg.matrix_rank(V1 + V2)
    return {
        "lambda": opt.x[0],
        "Q": opt.fun,
        "df": r - 1,
        "p_gof": scipy.stats.chi2.sf(opt.fun, df=r - 1),
        "lo": min(grid[keep]),
        "hi": max(grid[keep]),
        "bounded": not (keep[0] or keep[len(keep) - 1]),
    }


@app.cell
def _(idata_ase, idata_totals):
    # Measurment errors:
    beta_ase_draws = az.extract(idata_ase, "posterior")["beta"].to_numpy()
    beta_total_draws = az.extract(idata_totals, "posterior")["beta"].to_numpy()
    cov_ase = np.cov(beta_ase_draws)
    cov_total = np.cov(beta_total_draws)
    beta_ase = np.mean(beta_ase_draws, axis=1)
    beta_total = np.mean(beta_total_draws, axis=1)
    buffering_res = covariance_deming_regression(
        beta_ase, beta_total, cov_ase, cov_total
    )
    return beta_ase, beta_total, buffering_res, cov_ase, cov_total


@app.cell
def _(buffering_res):
    buffering_res
    return


@app.cell(hide_code=True)
def _(
    HAPLOTYPES,
    HAPLOTYPE_COLORS,
    beta_ase,
    beta_total,
    buffering_res,
    cov_ase,
    cov_total,
    lp,
    mo,
):
    def _():
        buff_factor = buffering_res["lambda"]
        df = pl.DataFrame(
            dict(
                hap=HAPLOTYPES,
                x=beta_ase,
                y=beta_total,
                x_se=np.sqrt(np.diag(cov_ase)),
                y_se=np.sqrt(np.diag(cov_total)),
            )
        ).with_columns(
            x_min=pl.col("x") - 1.96 * pl.col("x_se"),
            x_max=pl.col("x") + 1.96 * pl.col("x_se"),
            y_min=pl.col("y") - 1.96 * pl.col("y_se"),
            y_max=pl.col("y") + 1.96 * pl.col("y_se"),
        )
        xmin, xmax = min(df["x_min"]), max(df["x_max"])
        ymin, ymax = min(df["y_min"]), max(df["y_max"])
        lims = min(ymin, xmin), max(xmax, ymax)
        return (
            lp.ggplot(df, lp.aes("x", "y", color="hap"))
            + lp.geom_point(tooltips=lp.layer_tooltips(["hap"]))
            + lp.geom_errorbar(lp.aes(xmin="x_min", xmax="x_max"))
            + lp.geom_errorbar(lp.aes(ymin="y_min", ymax="y_max"))
            + lp.geom_point(
                data={"x": [0], "y": [0]},
                mapping=lp.aes("x", "y"),
                alpha=0,
                color="black",
            )  # ensure 0,0 is in view
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

    mo.vstack(
        [
            "Plot the ASE (binomial) model fit parameters versus the total counts (NB GLM) fit parameters. In dash red, the buffering fit line and in black the reference diagonal line. Note that the buffering fit line takes into account covariance of the estimate, which is typically substantial and means that the fit may not line between the points (it's intercept is always 0).",
            _(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # QC
    """)
    return


@app.cell(hide_code=True)
def _(
    all_compatibility_classes,
    allele_unique,
    gene_annot,
    gene_selector,
    mo,
    tx_annot,
):
    # Check multimapping across genes
    transcripts = list(tx_annot.filter(gene_id=gene_selector.value)["transcript_id"])

    def _():
        temp = []
        for mouse_id, emase in all_compatibility_classes.items():
            tx = np.isin(emase.lname, [tx.encode() for tx in transcripts])
            # 8 x n_classes array of compatibility with this gene
            gene_compat = np.array(
                [
                    sparse_any(scipy.sparse.csr_array(compat[tx, :]), axis=0)
                    for compat in emase.haps.values()
                ]
            )
            relevant_classes = sparse_any(gene_compat, axis=0)
            all_tx = np.array(
                [
                    sparse_any(
                        scipy.sparse.csr_array(compat[:, relevant_classes]), axis=1
                    )
                    for compat in emase.haps.values()
                ]
            ).any(axis=0)
            multimapper_tx = all_tx & (~tx)
            for mm_tx in np.where(multimapper_tx)[0]:
                mm_compat = (
                    np.array(
                        [
                            scipy.sparse.csr_array(compat)[mm_tx, :].todense()
                            for compat in emase.haps.values()
                        ]
                    ).any(axis=0)
                    & relevant_classes
                )
                temp.append(
                    {
                        "mouse_id": mouse_id,
                        "transcript_id": emase.lname[mm_tx].decode(),
                        "read_count": emase.count[mm_compat].sum(),
                    }
                )
        return pl.DataFrame(
            temp,
            schema={
                "mouse_id": pl.Utf8,
                "transcript_id": pl.Utf8,
                "read_count": pl.Int32,
            },
        )

    multimapper_tx_ids = _()
    multimapper_gene_ids = (
        multimapper_tx_ids.join(
            tx_annot.select("gene_id", "transcript_id"), "transcript_id"
        )
        .join(gene_annot.select("gene_id", "gene_name"), "gene_id")
        .join(
            allele_unique.filter(gene_id=gene_selector.value).select(
                "mouse_id", "diplotype"
            ),
            "mouse_id",
        )
        .sort("read_count")
    )
    mo.vstack(["List of gene multimapping reads", multimapper_gene_ids])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Utilities
    """)
    return


@app.cell
def _():
    # Get command-line arguments to work:
    import argparse

    parser = argparse.ArgumentParser("Run bias model on genes")
    parser.add_argument(
        "--gene_id",
        help="which gene to run on (optional)",
    )
    args = parser.parse_args()
    return (args,)


@app.cell
def _(ase_model_time, idata_ase, idata_totals, total_model_time):
    print(f"ASE model time: {ase_model_time:0.2f}")
    print(f"{idata_ase.posterior.attrs=}")
    print(f"Total model time: {total_model_time:0.2f}")
    print(f"{idata_totals.posterior.attrs=}")

    import os

    print(f"{os.cpu_count()=}")

    print(f"{len(os.sched_getaffinity(0))=}")

    from threadpoolctl import threadpool_info
    import pprint

    print("threadpool info:")
    pprint.pprint(threadpool_info())

    import numba

    print(f"{numba.get_num_threads()=}")
    return


@app.cell(disabled=True)
def _(ase_model):
    # Time dlogp
    wrapped_dlogp = ase_model.compile_dlogp(mode="numba")
    ip = ase_model.initial_point()

    # warm up
    wrapped_dlogp(ip)

    n = 100
    t = time.perf_counter()
    for _ in range(n):
        # logp, grad = f(x0)
        wrapped_dlogp(ip)
    dt = (time.perf_counter() - t) / n
    print(f"{dt*1e3:.2f} ms per dlogp  ->  1023 evals ≈ {dt*1023:.2f} s per draw")
    return


@app.cell
def _(all_compatibility_classes, mo, tx_annot):
    @mo.cache
    def get_gene_class_counts_cached(gene_id):
        return get_gene_class_counts(gene_id, tx_annot, all_compatibility_classes)

    return (get_gene_class_counts_cached,)


@app.cell
def _(HAPLOTYPES):
    def diplotypes_to_hapnums(diplotypes):
        HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}
        hapnums = diplotypes.select(
            hap1=pl.col("diplotype").str.slice(0, 1).replace_strict(HAP_TO_HAPNUM),
            hap2=pl.col("diplotype").str.slice(1, 1).replace_strict(HAP_TO_HAPNUM),
        )
        return hapnums["hap1"].to_numpy(), hapnums["hap2"].to_numpy()

    return


@app.cell(disabled=True)
def _(HAPLOTYPES, HAP_TO_HAPNUM, allele_unique, get_gene_class_counts_cached):
    # Relationship of variance and class prop in homozygous mice
    _gene_list = [
        x.strip()
        for x in """
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
    """.strip().splitlines()
    ]

    def _():
        temp = []
        for gene_id in _gene_list:
            gene_class_counts = get_gene_class_counts_cached(gene_id)
            gene_diplotypes = (
                allele_unique.filter(gene_id=gene_id)
                .select("diplotype", "mouse_id")
                .join(
                    pl.DataFrame({"mouse_id": gene_class_counts.ids}),
                    "mouse_id",
                    maintain_order="right",
                )
                .with_columns(
                    hap1=pl.col("diplotype")
                    .str.slice(0, 1)
                    .replace_strict(HAP_TO_HAPNUM),
                    hap2=pl.col("diplotype")
                    .str.slice(1, 1)
                    .replace_strict(HAP_TO_HAPNUM),
                )
            )
            q_hat = estimate_class_proportions(
                gene_class_counts.counts,
                gene_diplotypes["hap1"].to_numpy(),
                gene_diplotypes["hap2"].to_numpy(),
                len(HAPLOTYPES),
            )
            # NOTE: this works best if diplotypes are always sorted (AB not BA), which they are in
            # our dataset but may not be in general (e.g., simulated). Estimate is just noisier otherwise.
            for (diplotype, hap1, hap2), dat in gene_diplotypes.group_by(
                ["diplotype", "hap1", "hap2"]
            ):
                these_mice = dat["mouse_id"]
                these_mice_ids = np.where(np.isin(gene_class_counts.ids, these_mice))
                if len(these_mice) < 4:
                    continue
                these_counts = gene_class_counts.counts[these_mice_ids]
                tots = np.sum(these_counts, axis=1)
                q_hat_close = np.abs(np.log(q_hat[hap1] / q_hat[hap2])) < 0.5
                with np.errstate(divide="ignore", invalid="ignore"):
                    props = these_counts / np.sum(these_counts, axis=1, keepdims=True)
                    mprops = np.mean(props, axis=0)
                    mult_var = (
                        props * (1 - props) / these_counts.sum(axis=1, keepdims=True)
                    ).mean(axis=0)
                    true_var = props.var(axis=0, ddof=1)
                    excess_var = true_var - mult_var
                    alpha = mprops**2 / (true_var - mprops * np.mean(1 / tots))
                n_classes = gene_class_counts.compat.shape[0]
                temp.append(
                    pl.DataFrame(
                        {
                            "gene_id": gene_id,
                            "diplotype": diplotype,
                            "var": true_var,
                            "mean_T": np.mean(tots),
                            "mean_inv_T": np.mean(1 / tots),
                            "alpha": alpha,
                            "q_hat_close": q_hat_close,
                            "prop": mprops,
                            "n_samples": len(these_mice),
                            "class_num": np.arange(mprops.shape[0]),
                            "hap_str": [
                                "".join(
                                    h
                                    for i, h in enumerate(HAPLOTYPES)
                                    if gene_class_counts.compat[j, i]
                                )
                                for j in range(n_classes)
                            ],
                        }
                    )
                )
        df = pl.concat(temp)
        return df

    class_variance_df = _()
    return (class_variance_df,)


@app.cell
def _(HAPLOTYPES, class_variance_df, lp):
    _df = class_variance_df.filter(
        pl.col("prop") > 0.0,
        pl.col("prop").is_nan().not_(),
        pl.col("q_hat_close"),
        # pl.col("alpha") > 0,
    ).with_columns(
        alpha_nice=pl.when(pl.col("alpha") < 0).then(1e3).otherwise("alpha"),
        disp=1 / pl.col("alpha"),
    )
    _X = np.array(
        [
            np.ones(len(_df)),
            np.log(_df["prop"].to_numpy()),
            np.log(_df["mean_T"].to_numpy()),
        ]
    ).T
    res, _, _, _ = np.linalg.lstsq(_X, np.log(_df["var"].to_numpy()))
    _var = np.exp(
        res[0] + np.log(_df["prop"]) * res[1] + np.log(_df["mean_T"]) * res[2]
    )
    _fit = pl.DataFrame(
        {
            "prop": _df["prop"],
            "mean_T": _df["mean_T"],
            "gene_id": _df["gene_id"],
            "var": _var,
        }
    )
    print(res)
    (
        lp.ggplot(
            _df.with_columns(
                is_hom=pl.col("diplotype").is_in([f"{hap}{hap}" for hap in HAPLOTYPES]),
                # compatible = pl.col("diplotype").is_in(pl.col("hap_str").implode())
            ).sample(fraction=1.0),  # shuffle
            lp.aes("prop", "var", color="is_hom"),
        )
        + lp.facet_wrap("gene_id")
        + lp.geom_point(
            tooltips=lp.layer_tooltips(
                ["class_num", "n_samples", "diplotype", "hap_str"]
            ),
            alpha=0.3,
        )
        + lp.scale_x_log10()
        + lp.scale_y_continuous(trans="log10")
        + lp.geom_smooth(lp.aes("prop", "var"), inherit_aes=False, method="lm", deg=4)
        # + lp.geom_smooth(lp.aes("prop", "var"), inherit_aes=False, method="lm")
        + lp.geom_line(data=_fit, color="black")
    )
    return


@app.cell
def _(HAPLOTYPES, gene_class_counts, hap1, hap2):
    _q_hat = estimate_class_proportions(
        gene_class_counts.counts, hap1, hap2, len(HAPLOTYPES)
    )
    # q_hat
    # _hap1, _hap2
    var_fit, alpha_fit, _ = estimate_class_alpha(gene_class_counts, _q_hat, hap1, hap2)
    return


@app.cell(disabled=True)
def _(idata_ase):
    summary = az.summary(idata_ase)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pairwise plotter (ASE model)
    """)
    return


@app.cell
def _(idata_ase, mo):
    def extra_dims(var):
        """Dims of a variable other than chain/draw."""
        return [d for d in idata_ase.posterior[var].dims if d not in ("chain", "draw")]

    def coord_pickers(var):
        """One dropdown per extra dim of `var` (empty dict if scalar variable)."""
        da = idata_ase.posterior[var]
        return mo.ui.dictionary(
            {
                d: mo.ui.dropdown(
                    options={str(v): v for v in da[d].values},  # label -> value
                    value=str(da[d].values[0]),
                    label=d,
                )
                for d in extra_dims(var)
            }
        )

    plt_var_names = list(idata_ase.posterior.data_vars)
    return coord_pickers, plt_var_names


@app.cell
def _(mo, plt_var_names):
    x_var = mo.ui.dropdown(plt_var_names, value=plt_var_names[0], label="x variable")
    y_var = mo.ui.dropdown(plt_var_names, value=plt_var_names[-1], label="y variable")
    return x_var, y_var


@app.cell
def _(coord_pickers, x_var, y_var):
    x_coords = coord_pickers(x_var.value)
    y_coords = coord_pickers(y_var.value)
    return x_coords, y_coords


@app.cell(hide_code=True)
def _(mo, x_coords, x_var, y_coords, y_var):
    mo.vstack(
        [
            mo.hstack([x_var, x_coords], justify="start"),
            mo.hstack([y_var, y_coords], justify="start"),
        ]
    )
    return


@app.cell
def _(idata_ase, lp, x_coords, x_var, y_coords, y_var):
    def _():
        def select(var, coords):
            da = idata_ase.posterior[var].sel(**coords.value)  # -> (chain, draw)
            label = var + (
                "[" + ", ".join(str(v) for v in coords.value.values()) + "]"
                if coords.value
                else ""
            )
            return da, label

        xda, xlab = select(x_var.value, x_coords)
        yda, ylab = select(y_var.value, y_coords)

        df = pl.DataFrame(
            {
                "x": xda.values.ravel(),
                "y": yda.values.ravel(),
                "chain": [
                    str(c) for c in xda["chain"].values for _ in xda["draw"].values
                ],
            }
        )

        return (
            lp.ggplot(df, lp.aes("x", "y", color="chain"))
            + lp.geom_point(alpha=0.7, size=2.5)
            + lp.labs(x=xlab, y=ylab)
            + lp.ggsize(550, 450)
            + lp.ggtb()
        )

    _()
    return


@app.cell
def _(idata_ase, lp, mo):
    def _():
        x = idata_ase.posterior["q_logit"].mean(axis=(0, 1)).data
        hap = np.broadcast_to(
            idata_ase.posterior["q_logit"].coords["haplotypes"].data[:, None], x.shape
        ).astype(str)
        class_name = np.broadcast_to(
            idata_ase.posterior["q_logit"].coords["classes"].data[None, :], x.shape
        ).flatten()
        q_hat = idata_ase.constant_data["q_hat"].data
        pinned = idata_ase.constant_data["q_logit_pinned"].data
        return (
            lp.ggplot(
                pl.DataFrame(
                    {
                        "x": x.flatten(),
                        "y": x.flatten(),
                        "hap": hap.flatten(),
                        "q_hat": q_hat.flatten(),
                        "class_name": class_name,
                        "pinned": pinned.flatten(),
                    }
                ),
                lp.aes("q_hat", "y", color="hap", shape="pinned"),
            )
            + lp.geom_point(
                alpha=1.0, tooltips=lp.layer_tooltips(["hap", "class_name"])
            )
            + lp.scale_x_log10()
            + lp.labs(y="q_logit")
            + lp.ggtb()
        )
        # gene_class_counts.counts[hap1 == 0, 4]

    mo.vstack(
        [
            "Compare q_hat (naive estimate with no ASE) to the adjustments to q_hat (on a logit scale) that the full model makes",
            _(),
        ]
    )
    return


@app.cell
def _(gene_diplotypes, idata_ase, lp):
    def _():
        u = idata_ase.posterior["u"].mean(axis=(1)).data
        sample_ids = idata_ase.posterior.coords["samples"].data
        diplo = gene_diplotypes.join(
            pl.DataFrame({"mouse_id": sample_ids}),
            "mouse_id",
            how="right",
            maintain_order="right",
        )
        diplotype = np.broadcast_to(diplo["diplotype"], u.shape)
        chain = np.broadcast_to(np.arange(u.shape[0])[:, None], u.shape).astype(str)
        mouse_id = np.broadcast_to(diplo["mouse_id"], u.shape)
        return lp.ggplot(
            pl.DataFrame(
                {
                    "mouse_id": mouse_id.flatten(),
                    "diplotype": diplotype.flatten(),
                    "u": u.flatten(),
                    "chain": chain.flatten(),
                }
            ).sample(fraction=1.0),
            lp.aes("diplotype", "u", color="chain"),
        ) + lp.geom_point(
            alpha=0.5, tooltips=lp.layer_tooltips(["mouse_id", "diplotype", "chain"])
        )

    _()
    return


if __name__ == "__main__":
    app.run()
