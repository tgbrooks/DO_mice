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
    from ase_buffering.ase_model import make_ase_model
    from ase_buffering.total_counts_model import make_total_model

    config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
    HAPLOTYPES = config["haplotypes"].split(",")
    HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}
    SEX_TO_NUM = {"F": 0, "M": 1}


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
def _():
    OUTLIER_MOUSE_IDS = config["outlier_ids"]
    return (OUTLIER_MOUSE_IDS,)


@app.cell
def _(lp):
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
def _(allele_unique):
    diplotypes = allele_unique.select("mouse_id", "gene_id", "diplotype")
    return (diplotypes,)


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
def _(OUTLIER_MOUSE_IDS, mouse_ids):
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
        assert args.gene_id in gene_ids, (
            f"Provided gene id must be in gene ids (e.g. {gene_ids[:10]})"
        )
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


@app.cell
def _(all_compatibility_classes, diplotypes, gene_selector, tx_annot):
    gene_class_counts = get_gene_class_counts(
        gene_selector.value, tx_annot, all_compatibility_classes
    )
    ase_model = make_ase_model(gene_selector.value, gene_class_counts, diplotypes)
    return ase_model, gene_class_counts


@app.cell
def _(allele_unique, gene_class_counts, gene_selector):
    # For plotting
    n_samples = int(gene_class_counts.counts.shape[0])
    n_classes = int(gene_class_counts.compat.shape[0])
    n_haps = len(HAPLOTYPES)
    gene_diplotypes = (
        allele_unique.filter(gene_id=gene_selector.value)
        .select("diplotype", "mouse_id")
        .join(
            pl.DataFrame({"mouse_id": gene_class_counts.ids}),
            "mouse_id",
            maintain_order="right",
        )
    )
    hap1 = gene_diplotypes.select(
        pl.col("diplotype").str.slice(0, 1).replace_strict(HAP_TO_HAPNUM)
    )["diplotype"].to_numpy()
    hap2 = gene_diplotypes.select(
        pl.col("diplotype").str.slice(1, 1).replace_strict(HAP_TO_HAPNUM)
    )["diplotype"].to_numpy()
    return hap1, hap2, n_classes, n_haps


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
            random_seed=101,
            nuts_sampler="nutpie",
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
        (idata_ase["posterior"]["sigma_gamma"] * idata_ase["posterior"]["gamma"])
        .to_numpy()
        .reshape((-1, (n_haps * (n_haps - 1) // 2)))
    )
    _df = (
        pl.DataFrame({f"gamma_{i}": _gamma[:, i] for i in range(_gamma.shape[1])})
        .unpivot(on=[f"gamma_{i}" for i in range(_gamma.shape[1])])
        .group_by("variable")
        .agg(
            mean=pl.col("value").mean(),
            low=pl.col("value").quantile(0.05),
            hi=pl.col("value").quantile(0.95),
        )
    )
    (
        lp.ggplot(_df, lp.aes(x="mean", y="variable"))
        + lp.geom_point()
        + lp.geom_errorbar(lp.aes(xmin="low", xmax="hi"))
        + lp.labs(x="gamma")
    )
    return


@app.cell
def _(idata_ase):
    az.summary(idata_ase, var_names="beta", filter_vars="like")
    return


@app.cell(hide_code=True)
def _(gene_class_counts, idata_ase, lp, n_classes):
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
    class_summaries = (
        idata_ase["posterior"]["q"].values @ gene_class_counts.compat
    ).mean(axis=(0, 1))
    leak_outs = np.array(
        [
            (
                idata_ase["posterior"]["q"].values[:, :, i, :]
                @ ((~gene_class_counts.compat[:, i, None]) & gene_class_counts.compat)
            ).mean(axis=(0, 1))
            for i in range(len(HAPLOTYPES))
        ]
    )
    max_leak = leak_outs.max()
    class_summaries = pl.concat(
        [
            pl.DataFrame(
                {
                    "source_hap": hap,
                    "out_class": class_summaries[i],
                    "leak_to": leak_outs[i],
                    "out_hap": HAPLOTYPES,
                }
            )
            for i, hap in enumerate(HAPLOTYPES)
        ]
    )
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
def _(gene_class_counts, hap1, hap2, idata_ase, lp, mo):
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
def _(gene_class_counts, hap1, hap2, idata_ase, lp, n_classes):
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
def _(HAPLOTYPE_COLORS, gene_class_counts, idata_ase, lp):
    # Plot fit allele-unique bias rates
    def _():
        q = idata_ase["posterior"]["q"].values
        temp = []
        for i, h1 in enumerate(HAPLOTYPES):
            for j, h2 in enumerate(HAPLOTYPES):
                h1_unique_classes = gene_class_counts.compat[:, i] & (
                    ~gene_class_counts.compat[:, j]
                )
                h1_u = q[..., i, h1_unique_classes].sum(axis=-1).mean(axis=(0, 1))

                h2_unique_classes = gene_class_counts.compat[:, j] & (
                    ~gene_class_counts.compat[:, i]
                )
                h2_u = q[..., j, h2_unique_classes].sum(axis=-1).mean(axis=(0, 1))
                temp.append(
                    {
                        "source_hap": h1,
                        "other_hap": h2,
                        "frac_unique": h1_u,
                        "reverse_frac_unique": h2_u,
                        "allele_unique_bias": h1_u / h2_u,
                    }
                )
        return pl.DataFrame(temp).filter(pl.col("source_hap") != pl.col("other_hap"))

    _df = _()
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
    total_model = make_total_model(gene_selector.value, gene_class_counts, diplotypes, all_pheno)
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


if __name__ == "__main__":
    app.run()
