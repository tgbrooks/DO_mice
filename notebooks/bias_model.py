import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import lets_plot as lp
    import polars as pl
    import polars_bio as pb
    import numpy as np
    import yaml
    import scipy.sparse
    import scipy.optimize
    import scipy.stats
    from dataclasses import dataclass
    import pytensor
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    import time

    from util.compressed_emase import load_compressed_emase

    return (
        az,
        dataclass,
        load_compressed_emase,
        lp,
        mo,
        np,
        pb,
        pl,
        pm,
        pt,
        scipy,
        time,
        yaml,
    )


@app.cell
def _(pl):
    size_factors = pl.read_csv(
        "processed/Adipose/size_factors.txt", separator="\t"
    )
    return (size_factors,)


@app.cell
def _(lp, yaml):
    config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
    HAPLOTYPES = config["haplotypes"].split(",")
    HAPLOTYPE_COLORS = lp.scale_color_brewer(palette="Dark2").palette(
        len(HAPLOTYPES)
    )
    OUTLIER_MOUSE_IDS = config["outlier_ids"]
    return HAPLOTYPES, HAPLOTYPE_COLORS, OUTLIER_MOUSE_IDS


@app.cell
def _(HAPLOTYPES):
    HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}
    return (HAP_TO_HAPNUM,)


@app.cell
def _():
    SEX_TO_NUM = {"F": 0, "M": 1}
    return (SEX_TO_NUM,)


@app.cell
def _(pb, pl):
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
def _(pb, pl):
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
def _(pl):
    allele_unique = pl.read_parquet(
        "processed/Adipose/allele_unique_reads.parquet"
    )
    allele_unique
    return (allele_unique,)


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
def _(HAPLOTYPES, OUTLIER_MOUSE_IDS, load_compressed_emase, mouse_ids):
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
def _(allele_unique, args, gene_annot, mo, pl):
    _vals = (
        allele_unique.select("gene_id")
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
    if args.gene_id:
        assert args.gene_id in gene_ids, (
            f"Provided gene id must be in gene ids (e.g. {gene_ids[:10]})"
        )
        (default,) = [n for n, id in _options.items() if id == args.gene_id]
    else:
        default = None
    gene_selector = mo.ui.dropdown(_options, value=default, searchable=True)
    gene_selector
    return (gene_selector,)


@app.cell
def _(
    all_compatibility_classes,
    dataclass,
    gene_selector,
    np,
    out,
    scipy,
    tx_annot,
):
    # select just this genes data
    transcripts = list(
        tx_annot.filter(gene_id=gene_selector.value)["transcript_id"]
    )

    @dataclass
    class CompatClasses:
        compat: np.ndarray  # 8 x n_classes - compatibility of the read class with each of the 8 haplotypes
        counts: (
            np.ndarray
        )  # n_classes - number of reads of this compatibility class

    def _any(sparse_matrix, axis):
        return (sparse_matrix != 0).sum(axis=axis) > 0

    # First get data for just one gene
    def extract_gene(emase, transcripts):
        tx = np.isin(emase.lname, [tx.encode() for tx in transcripts])
        # 8 x n_classes array of compatibility with this gene
        gene_compat = np.array(
            [
                _any(scipy.sparse.csr_array(compat[tx, :]), axis=0)
                for compat in emase.haps.values()
            ]
        )
        relevant_classes = _any(gene_compat, axis=0)
        final_compat = gene_compat[:, relevant_classes]
        counts = emase.count[relevant_classes]
        # Note: classes arise from *transcripts* not genes and therefore could have identical gene compatibility
        # in different classes. We'll combine those in the next step.
        return CompatClasses(compat=final_compat, counts=counts)

    compatibility_classes = {
        mouse_id: extract_gene(emase, transcripts)
        for mouse_id, emase in all_compatibility_classes.items()
    }

    @dataclass
    class CompatClassesDf:
        """All read compatibility data from one gene across all samples"""

        compat: np.ndarray  # 8 x n_classes
        counts: np.ndarray  # n_samples x n_classes
        ids: list[str]  # n_samples ids list

    # Now uniformize it: all samples to have the *same* compatibility classes
    def uniformize_compat_classes(
        compat_classes: dict[str, CompatClasses], min_n_classes: int | None
    ) -> CompatClassesDf:
        """
        Aggregate classes and counts, putting zeros in classes that are missing for any sample.
        Pad classes to min_n_classes.
        """
        n_samples = len(compat_classes)
        classes = set()
        for compat in compat_classes.values():
            classes.update([tuple(col) for col in compat.compat.T])
        classes = np.array(
            [np.array(cls) for cls in classes]
        )  # fix the ordering
        if min_n_classes is not None and classes.shape[0] < min_n_classes:
            # Pad with classes that incompatible with all haplotypes
            # These should have no reads reported since such reads simply don't align to this gene
            n_padding_classes = min_n_classes - len(classes)
            classes = np.concatenate(
                [
                    classes,
                    np.zeros(
                        (n_padding_classes, classes.shape[1]), dtype=bool
                    ),
                ]
            )
        compat_map = {tuple(row): i for i, row in enumerate(classes)}
        counts_out = np.zeros((n_samples, classes.shape[0]))
        for i, compat in enumerate(compat_classes.values()):
            matches = np.array(
                [compat_map[tuple(col)] for col in compat.compat.T]
            )
            # matches may have repeated indices due to transcripts classes being identical at gene level
            # we sum those repeated indices
            np.add.at(counts_out[i], matches, compat.counts)
            # counts_out[i, matches] += compat.counts
        return CompatClassesDf(
            compat=classes,
            counts=counts_out,
            ids=compat_classes.keys(),
        )
        return out

    gene_class_counts = uniformize_compat_classes(
        compatibility_classes, min_n_classes=None
    )
    return (gene_class_counts,)


@app.cell
def _(
    HAPLOTYPES,
    HAP_TO_HAPNUM,
    allele_unique,
    gene_class_counts,
    gene_selector,
    np,
    pl,
    pm,
    pt,
):
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
    n_samples = int(gene_class_counts.counts.shape[0])
    n_classes = int(gene_class_counts.compat.shape[0])
    n_haps = len(HAPLOTYPES)
    obs_total_counts = gene_class_counts.counts.sum(axis=1)
    model = pm.Model(
        coords={
            "haplotypes": HAPLOTYPES,
            "classes": [f"class_{i}" for i in range(n_classes)],
            "samples": gene_class_counts.ids,
        }
    )
    DIRICHLET_PRIOR = (
        1 / 5
    )  # Encourages very low proportions for unobserved classes
    # mask out classes which have no expression from anything with that haplotype
    mask = np.array(
        [
            gene_class_counts.counts[hap1 == i].any(axis=0).astype(int)
            | gene_class_counts.counts[hap2 == i].any(axis=0).astype(int)
            for i in range(len(HAPLOTYPES))
        ]
    )
    with model:
        _counts = pm.Data(
            "counts", gene_class_counts.counts, dims=("samples", "classes")
        )
        _hap1 = pm.Data("hap1", hap1, dims="samples")
        _hap2 = pm.Data("hap2", hap2, dims="samples")
        _mask = pm.Data("mask", mask, dims=("haplotypes", "classes"))
        _nz = _counts > 0

        # Nuisance variables
        # Rates at which reads from a haplotype are assigned to each class
        # q = pm.Dirichlet('q', a=np.ones(n_classes)*DIRICHLET_PRIOR, dims=("haplotypes", "classes"))
        q_raw = pm.Normal("q_raw", sigma=3, dims=("haplotypes", "classes"))

        def masked_softmax(q, mask):
            exp = pm.math.exp(q) * mask
            norm = exp.sum(axis=1)[:, None]
            return exp / norm

        q = pm.Deterministic(
            "q", masked_softmax(q_raw, mask), dims=("haplotypes", "classes")
        )

        # Actual effects
        beta = pm.ZeroSumNormal("beta", sigma=2.0, dims="haplotypes")

        # Random per-sample effects
        sigma_u = pm.HalfNormal("sigma_u", 1.0)
        u_raw = pm.Normal("u_raw", 0, 1, dims="samples")
        u = sigma_u * u_raw

        p = pm.Deterministic(
            "p",
            pm.math.sigmoid(beta[_hap1] - beta[_hap2] + u),
            dims=("samples"),
        )
        class_props = p[:, None] * q[_hap1] + (1 - p)[:, None] * q[_hap2]

        # Log-likelihood - used instaed of pm.Multinomial since its a bit faster
        # We drop the constant terms that don't depend upon class_props
        pm.Potential("ll", pt.sum(_counts[_nz] * pt.log(class_props[_nz])))

        # pm.Multinomial("class_counts", n=obs_total_counts, p=class_props,
        #               observed=gene_class_counts.counts, dims=("samples","classes"))
    return hap1, hap2, model, n_classes


@app.cell
def _(mo, model, pm):
    mo.mermaid(pm.model_to_mermaid(model))
    return


@app.cell
def _(args, model, pm, time):
    # Fit the model
    # with pytensor.config.change_flags(profile=True, profiling__time_thunks = True):
    _start = time.time()
    with model:
        idata_ase = pm.sample(
            300,
            random_seed=102,
            nuts_sampler="nutpie",
            backend=args.backend,
            # nuts={"low_rank_modified_mass_matrix": True}
        )
    _end = time.time()
    ase_model_time = _end - _start
    return ase_model_time, idata_ase


@app.cell
def _(az, idata_ase):
    az.plot_trace_dist(idata_ase, var_names=["beta"], compact=False)
    return


@app.cell
def _(az, idata_ase):
    az.plot_forest(
        idata_ase,
        var_names=["beta"],
        combined=True,
        ci_probs=[0.5, 0.95],
        figure_kwargs={"figsize": (10, 3)},
    )
    return


@app.cell
def _(az, idata_ase):
    az.summary(idata_ase, var_names="beta", filter_vars="like")
    return


app._unparsable_cell(
    r"""
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
        y = pl.col("q_hat")
        ymin=pl.col("q_hat") - 1.96 * pl.col("q_hat_se"),
        ymax=pl.col("q_hat") + 1.96 * pl.col("q_hat_se"),
    )
    (
        lp.ggplot(_dat, lp.aes(x="class", y="q_hat", color="is_leak"))
        + lp.facet_wrap("haplotype")
        + lp.geom_point(
            tooltips=lp.layer_tooltips(
                ["haplotype", "class", "q_hat", "ymin", "ymax", "compat_with"]
            )
        )
        + lp.geom_errorbar(lp.aes(ymin="ymin", ymax="ymax"))
        + lp.scale_color_manual(breaks=[False, True], values=["black", "red"])
        + lp.ggtb()
        + lp.ggtitle("Class proportions by haplotype")
    )
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _(HAPLOTYPES, gene_class_counts, idata_ase, lp, np, pl):
    class_summaries = (
        idata_ase["posterior"]["q"].values @ gene_class_counts.compat
    ).mean(axis=(0, 1))
    leak_outs = np.array(
        [
            (
                idata_ase["posterior"]["q"].values[:, :, i, :]
                @ (
                    (~gene_class_counts.compat[:, i, None])
                    & gene_class_counts.compat
                )
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


@app.cell
def _(HAPLOTYPES, gene_class_counts, hap1, hap2, idata_ase, lp, mo, np, pl):
    # For each diplotype, plot the fit percentage of reads that are allele-unique for both haplotypes
    def _():
        temp = []
        p = idata_ase["posterior"]["p"].values
        q = idata_ase["posterior"]["q"].values
        class_props = (
            p[:, :, :, None] * q[:, :, hap1]
            + (1 - p)[:, :, :, None] * q[:, :, hap2]
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
                    gene_class_counts.counts[selected][
                        :, h1_unique_classes
                    ].sum(axis=1)
                    / tots
                )
                actual_h2_unique_counts = (
                    gene_class_counts.counts[selected][
                        :, h2_unique_classes
                    ].sum(axis=1)
                    / tots
                )
                pred_allele_ratio = np.empty(p.shape[2])
                pred_allele_ratio[selected1] = p[:,:,selected1].mean(axis=(0,1))
                pred_allele_ratio[selected2] = 1-p[:,:,selected2].mean(axis=(0,1))
                actual_allele_ratio = actual_h1_unique_counts / (actual_h2_unique_counts + actual_h1_unique_counts)
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
    _res = (
        pred_allele_unique
        .unpivot(
            index=["hap1", "hap2"],
            on=[
                "pred_hap1_unique_counts",
                "actual_hap1_unique_counts",
                # Only show hap1 values - the lower/upper triangular show opposite haps
                #"pred_hap2_unique_counts",
                #"actual_hap2_unique_counts",
            ],
            variable_name="var",
            value_name="frac",
        )
        .with_columns(
            type=pl.col("var").str.split("_").list.get(0),
            hap=pl.col("var").str.split("_").list.get(1),
        )
    )
    _res2 = (
        pred_allele_unique
        .unpivot(
            index=["hap1", "hap2"],
            on=["pred_allele_ratio", "actual_allele_ratio"],
            variable_name="var",
            value_name="allele_ratio",
        )
        .with_columns(
            type=pl.col("var").str.split("_").list.get(0),
        )
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


@app.cell
def _(
    HAPLOTYPES,
    gene_class_counts,
    hap1,
    hap2,
    idata_ase,
    lp,
    n_classes,
    np,
    pl,
):
    def _():
        p = idata_ase["posterior"]["p"].values
        q = idata_ase["posterior"]["q"].values
        class_props = (
            p[:, :, :, None] * q[:, :, hap1]
            + (1 - p)[:, :, :, None] * q[:, :, hap2]
        ).mean(axis=(0,1))
        # Compare predicted class props to homozygous class props
        # Homozygous directly informs class props without any inference of proportions
        # so we expect these to be close if there are a good number of homozygous
        homo = (hap1 == hap2)
        temp = []
        for i in range(n_classes):
            temp.append(pl.DataFrame({
                "mouse_id": np.array(list(gene_class_counts.ids))[homo],
                "hap": np.array(HAPLOTYPES)[hap1[homo]],
                "class": i,
                "actual_prop": gene_class_counts.counts[homo,i] / gene_class_counts.counts[homo,:].sum(axis=1),
                "pred_prop": class_props[homo,i],
            }))
        df = pl.concat(temp)
        return df
    _compat_with = {
        k: "".join(
                hap2
                for j, hap2 in enumerate(HAPLOTYPES)
                if gene_class_counts.compat[k, j]
            )
            for k in range(n_classes)
    }
    _df = (
        _()
        .group_by("hap", "class")
        .agg(
            pl.col('actual_prop').mean(),
            pl.col('pred_prop').mean(),
        )
        .with_columns(
            pl.col("class").replace_strict(_compat_with, return_dtype=str),
        )
    )
    (
        lp.ggplot(
            _df,
            lp.aes("actual_prop", "pred_prop", color="class")
        )
        + lp.geom_point()
        + lp.geom_abline(intercept=1.0, slope=1, color="black", linestyle=2)
        + lp.scale_x_log10()
        + lp.scale_y_log10()
        + lp.facet_wrap("hap")
        + lp.labs(x="actual prop (in homozygous)", y="predicted prop")
        + lp.ggtb()
        + lp.ggtitle("Alignment classes in homozygous versus predicted")
    )
    return


@app.cell
def _(HAPLOTYPES, HAPLOTYPE_COLORS, gene_class_counts, idata_ase, lp, pl):
    # Plot fit allele-unique bias rates
    def _():
        q = idata_ase["posterior"]["q"].values
        temp = []
        for i, h1 in enumerate(HAPLOTYPES):
            for j, h2 in enumerate(HAPLOTYPES):
                h1_unique_classes = gene_class_counts.compat[:, i] & (
                    ~gene_class_counts.compat[:, j]
                )
                h1_u = q[...,i,h1_unique_classes].sum(axis=-1).mean(axis=(0,1))

                # For comparison, the
                h2_unique_classes = gene_class_counts.compat[:, j] & (
                    ~gene_class_counts.compat[:, i]
                )
                h2_u = q[...,j,h2_unique_classes].sum(axis=-1).mean(axis=(0,1))
                temp.append({
                    "source_hap": h1,
                    "other_hap": h2,
                    "frac_unique": h1_u,
                    "reverse_frac_unique": h2_u,
                    "allele_unique_bias": h1_u / h2_u,
                })
        return pl.DataFrame(temp).filter(pl.col("source_hap") != pl.col("other_hap"))
    _df = _()
    (
        lp.ggplot(
            _df,
            lp.aes(x="frac_unique", y="reverse_frac_unique", color="source_hap")
        )
        + lp.geom_point(tooltips=lp.layer_tooltips(["source_hap", "other_hap", "frac_unique", "reverse_frac_unique", "allele_unique_bias"]))
        + lp.geom_abline(intercept=0,slope=1)
        + lp.scale_color_manual(breaks=HAPLOTYPES, values=HAPLOTYPE_COLORS)
        + lp.ggtitle("Biases in rate of producing allele unique reads")
    )

    return


@app.cell
def _(az, idata_ase):
    az.plot_energy(idata_ase)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Total counts model
    """)
    return


@app.cell
def _(allele_unique, gene_class_counts, gene_selector, pl):
    gene_totals = pl.DataFrame(
        {
            "mouse_id": gene_class_counts.ids,
            "total_reads": gene_class_counts.counts.sum(axis=1),
        }
    ).join(
        allele_unique.filter(gene_id=gene_selector.value).select(
            "mouse_id", "diplotype"
        ),
        "mouse_id",
    )
    gene_totals
    return (gene_totals,)


@app.cell
def _(gene_totals, pl, size_factors):
    # phenotypes were acquired from Dryad  and extracted from Rdata format:
    # https://datadryad.org/dataset/doi:10.5061/dryad.pj105
    pheno = (
        pl.read_csv("phenotypes.csv.gz")
        .rename({"mouse.id": "mouse_id"})
        .with_columns(
            pl.col("DOwave")
            .cast(str)
            .cast(pl.Enum([str(x) for x in range(1, 6)]))
        )
        .join(size_factors, "mouse_id")
        .join(
            gene_totals.select("mouse_id"), "mouse_id", maintain_order="right"
        )
    )
    return (pheno,)


@app.cell
def _(gene_annot, gene_selector, gene_totals, np, pl):
    _chrom = gene_annot.filter(gene_id=gene_selector.value)["chrom"][0]
    kinship = gene_totals.select("mouse_id").join(
        pl.read_csv(f"geno/kinship/{_chrom}.txt", separator="\t").select(
            "mouse_id", *gene_totals["mouse_id"]
        ),
        "mouse_id",
        maintain_order="left",
    )
    assert (kinship["mouse_id"] == kinship.columns[1:]).all()
    kinship_eigs = np.linalg.eigh(kinship.drop("mouse_id").to_numpy())
    return


@app.cell
def _(
    HAPLOTYPES,
    HAP_TO_HAPNUM,
    SEX_TO_NUM,
    gene_class_counts,
    gene_totals,
    np,
    pheno,
    pl,
    pm,
):
    def _():
        total_model = pm.Model(
            coords={
                "haplotypes": HAPLOTYPES,
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
            beta_wave = pm.ZeroSumNormal(
                "beta_wave", sigma=0.5, dims="DOwaves"
            )
            sigma_u = pm.HalfNormal("sigma_u", sigma=1)
            # h = pm.Beta("h", alpha=1, beta=1) # heritability
            # Compute square-root of sigma_u^2 (hK + (1-h)I) using K = QVQ^T, I = QQ^T
            # sample_covar = sigma_u * kinship_eigs.eigenvectors * pt.sqrt(h * kinship_eigs.eigenvalues + (1-h))
            # u_kinship = sample_covar @ pm.Normal("u_kinship", dims="samples")
            # u_kinship = sigma_u * pm.Normal("u_kinship", dims="samples")
            u_kinship = 0
            log_disp = pm.Normal("log_disp", mu=np.log(0.01), sigma=2)
            mu = pm.Deterministic(
                "mu",
                (pm.math.exp(beta[hap1]) + pm.math.exp(beta[hap2]))
                * pm.math.exp(
                    intercept
                    + sex * beta_M
                    + beta_wave[DOwave]
                    + u_kinship
                    + sf
                ),
                dims="samples",
            )
            total_reads = pm.NegativeBinomial(
                "total_reads",
                mu=mu,
                alpha=1 / np.exp(log_disp),
                observed=gene_totals["total_reads"].to_numpy(),
                dims="samples",
            )
        return total_model

    total_model = _()
    return (total_model,)


@app.cell
def _(mo, pm, total_model):
    mo.mermaid(pm.model_to_mermaid(total_model))
    return


@app.cell
def _(pm, time, total_model):
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
def _(az, idata_totals):
    az.plot_forest(
        idata_totals,
        var_names=["beta"],
        combined=True,
        ci_probs=[0.5, 0.95],
        figure_kwargs={"figsize": (10, 3)},
    )
    return


@app.cell
def _(az, idata_totals, pl):
    pl.DataFrame(
        az.summary(idata_totals, ["beta"], filter_vars="like").reset_index()
    )
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


@app.cell
def _(np, scipy):
    def covariance_deming_regression(
        b1, b2, V1, V2, grid=np.linspace(-2, 3, 2001)
    ):
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
            k = (
                e.eigenvalues > np.max(e.eigenvalues) * 1e-8
            )  # generalized inverse
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

    return (covariance_deming_regression,)


@app.cell
def _(az, covariance_deming_regression, idata_ase, idata_totals, np):
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


@app.cell
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
    np,
    pl,
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
            + lp.labs(
                x="beta (ASE, binomial GLM)", y="beta (total counts, NB GLM)"
            )
            + lp.scale_color_manual(
                values=HAPLOTYPE_COLORS,
                breaks=HAPLOTYPES,
            )
            + lp.geom_abline(
                slope=buff_factor, intercept=0, color="red", linetype=2
            )
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
    parser.add_argument(
        "--backend",
        help="which PYMC backend to use (numba or jax)",
        default="numba",
    )
    args = parser.parse_args()
    return (args,)


@app.cell
def _(args, ase_model_time, idata_ase, idata_totals, total_model_time):
    print(f"PYMC backend = {args.backend}")
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

    import jax

    # Print the active default backend ('cpu', 'gpu', or 'tpu')
    print("JAX default backend:", jax.default_backend())
    print("Available devices:", jax.devices())
    return


@app.cell(disabled=True)
def _(model):
    import timeit

    f_numba = model.compile_dlogp(mode="NUMBA")
    f_jax = model.compile_dlogp(mode="JAX")
    ip = model.initial_point()

    f_numba(ip)
    f_jax(ip)  # warm up: JIT + XLA compile
    print(timeit.timeit(lambda: f_numba(ip), number=1000) / 1000)
    print(timeit.timeit(lambda: f_jax(ip), number=1000) / 1000)
    return


if __name__ == "__main__":
    app.run()
