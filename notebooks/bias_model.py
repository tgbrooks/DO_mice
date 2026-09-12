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
    from dataclasses import dataclass
    import pytensor
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt

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
        pytensor,
        scipy,
        yaml,
    )


@app.cell
def _(pl):
    size_factors = pl.read_csv("processed/Adipose/size_factors.txt", separator="\t")
    return


@app.cell
def _(lp, yaml):
    config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
    HAPLOTYPES = config["haplotypes"].split(",")
    HAPLOTYPE_COLORS = lp.scale_color_brewer(palette="Dark2").palette( len(HAPLOTYPES) )
    OUTLIER_MOUSE_IDS = config['outlier_ids']
    return HAPLOTYPES, OUTLIER_MOUSE_IDS


@app.cell
def _(HAPLOTYPES):
    HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}
    return (HAP_TO_HAPNUM,)


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
    allele_unique = pl.read_parquet("processed/Adipose/allele_unique_reads.parquet")
    allele_unique
    return (allele_unique,)


@app.cell
def _(allele_unique):
    mouse_ids = sorted(allele_unique['mouse_id'].unique())
    return (mouse_ids,)


@app.cell
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
    Then each sample (and for each gene), we observe the number of reads that are compatible with
    """)
    return


@app.cell
def _(HAPLOTYPES, OUTLIER_MOUSE_IDS, load_compressed_emase, mouse_ids):
    # Load ALL the read compatibility data form all the mice
    all_compatibility_classes = {}
    for mouse_id in mouse_ids:
        if mouse_id in OUTLIER_MOUSE_IDS:
            continue
        all_compatibility_classes[mouse_id] = load_compressed_emase(f"processed/Adipose/gbrs/{mouse_id}.compressed.h5", [f'h{i}' for i in range(len(HAPLOTYPES))])
    return (all_compatibility_classes,)


@app.cell(hide_code=True)
def _(allele_unique, gene_annot, mo, pl):
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
    gene_selector = mo.ui.dropdown(_options, searchable=True)
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
    transcripts = list(tx_annot.filter(gene_id = gene_selector.value)['transcript_id'])
    @dataclass
    class CompatClasses:
        compat: np.ndarray # 8 x n_classes - compatibility of the read class with each of the 8 haplotypes
        counts: np.ndarray # n_classes - number of reads of this compatibility class
    def _any(sparse_matrix, axis):
        return ((sparse_matrix != 0).sum(axis=axis) > 0)

    # First get data for just one gene
    def extract_gene(emase, transcripts):
        tx = np.isin(emase.lname, [tx.encode() for tx in transcripts])
        # 8 x n_classes array of compatibility with this gene
        gene_compat = np.array([_any(scipy.sparse.csr_array(compat[tx,:]), axis=0) for compat in emase.haps.values()])
        relevant_classes = _any(gene_compat, axis=0)
        final_compat = gene_compat[:, relevant_classes]
        counts = emase.count[relevant_classes]
        # Note: classes arise from *transcripts* not genes and therefore could have identical gene compatibility
        # in different classes. We'll combine those in the next step.
        return CompatClasses(compat=final_compat, counts=counts)
    compatibility_classes = {mouse_id: extract_gene(emase, transcripts) for mouse_id, emase in all_compatibility_classes.items()}

    @dataclass
    class CompatClassesDf:
        """ All read compatibility data from one gene across all samples """
        compat: np.ndarray # 8 x n_classes
        counts: np.ndarray # n_samples x n_classes
        ids: list[str] # n_samples ids list

    # Now uniformize it: all samples to have the *same* compatibility classes
    def uniformize_compat_classes(compat_classes: dict[str,CompatClasses]) -> CompatClassesDf:
        n_samples = len(compat_classes)
        classes = set()
        for compat in compat_classes.values():
            classes.update([tuple(col) for col in compat.compat.T])
        classes = np.array([np.array(cls) for cls in classes]) # fix the ordering
        compat_map = {tuple(row): i for i, row in enumerate(classes)}
        counts_out = np.zeros((n_samples, classes.shape[0]))
        for i, compat in enumerate(compat_classes.values()):
            matches = np.array([compat_map[tuple(col)] for col in compat.compat.T])
            # matches may have repeated indices due to transcripts classes being identical at gene level
            # we sum those repeated indices
            np.add.at(counts_out[i], matches, compat.counts)
            #counts_out[i, matches] += compat.counts
        return CompatClassesDf(
            compat = classes,
            counts = counts_out,
            ids = compat_classes.keys(),
        )
        return out
    gene_class_counts = uniformize_compat_classes(compatibility_classes)
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
    gene_diplotypes = allele_unique.filter(gene_id = gene_selector.value).select('diplotype', 'mouse_id').join(pl.DataFrame({"mouse_id": gene_class_counts.ids}), "mouse_id")
    hap1 = gene_diplotypes.select(pl.col("diplotype").str.slice(0,1).replace_strict(HAP_TO_HAPNUM))['diplotype'].to_numpy()
    hap2 = gene_diplotypes.select(pl.col("diplotype").str.slice(1,1).replace_strict(HAP_TO_HAPNUM))['diplotype'].to_numpy()
    n_samples = gene_class_counts.counts.shape[0]
    n_classes = gene_class_counts.compat.shape[0]
    n_haps = len(HAPLOTYPES)
    obs_total_counts = gene_class_counts.counts.sum(axis=1)
    model = pm.Model(
        coords = {
            "haplotypes": HAPLOTYPES,
            "classes": [f'class_{i}' for i in range(n_classes)],
            "samples": gene_class_counts.ids,
        }
    )
    #valid = gene_class_counts.compat.T.astype(bool)   # (n_haps, n_classes)
    with model:
        # one Dirichlet per haplotype, over only its valid classes
        q = pm.Dirichlet('q', a=np.ones(n_classes), dims=("haplotypes", "classes"))
        #eps = pm.Beta("eps", 2, 100, dims="haplotypes")
        #q_rows = []
        #for g in range(n_haps):
        #    idx = np.flatnonzero(valid[g])
        #    row = pm.Dirichlet(f"q_{g}", a=np.ones(len(idx)))
        #    leak_idx = np.flatnonzero(~valid[g])
        #    row_leak = pm.Dirichlet(f"q_leak_{g}", a=np.ones(len(leak_idx)))
        #    vals = pt.zeros(n_classes)
        #    vals = pt.set_subtensor(vals[idx], row * (1-eps[g]))
        #    vals = pt.set_subtensor(vals[leak_idx], row_leak * eps[g])
        #    q_rows.append(vals)
        #q = pm.Deterministic("q", pt.stack(q_rows), dims=("haplotypes", "classes"))
        #q_rows = []
        #for hap in range(n_haps):
        #    _c = (gene_class_counts.counts[hap1 == hap].mean(axis=0) + gene_class_counts.counts[hap2 == hap].mean(axis=0))
        #    q_rows.append(_c / _c.sum())
        #q = pt.stack(q_rows)

        # Actual effects
        beta = pm.ZeroSumNormal("beta", sigma=2.0, dims="haplotypes")

        # Random per-sample effects
        sigma_u = pm.HalfNormal("sigma_u", 1.0)
        u_raw = pm.Normal("u_raw", 0, 1, dims="samples")

        p = pm.Deterministic("p", pm.math.sigmoid(beta[hap1] - beta[hap2] + sigma_u * u_raw), dims=("samples"))
        class_props = p[:, None] * q[hap1] + (1 - p)[:, None] * q[hap2]

        # Log-likelihood - used instaed of pm.Multinomial since its a bit faster
        # We drop the constant terms that don't depend upon class_props
        counts_const = pt.as_tensor_variable(gene_class_counts.counts)
        pm.Potential("ll", pt.sum(counts_const * pt.log(class_props)))

        #pm.Multinomial("class_counts", n=obs_total_counts, p=class_props,
        #               observed=gene_class_counts.counts, dims=("samples","classes"))
    return (model,)


@app.cell
def _(mo, model, pm):
    mo.mermaid(pm.model_to_mermaid(model))
    return


@app.cell
def _(model, pm):
    # Fit the model
    #with pytensor.config.change_flags(profile=True, profiling__time_thunks = True):
    with model:
        samples = pm.sample(
            300,
            random_seed=100,
            #target_accept=0.6,
            nuts_sampler="nutpie",
            #nuts_sampler_kwargs={"low_rank_modified_mass_matrix": True}
        )
    return (samples,)


@app.cell
def _(samples):
    samples.sample_stats['depth'].mean()
    return


@app.cell
def _(az, samples):
    t = samples.posterior.sampling_time      # or time it yourself
    print(az.ess(samples, var_names=["beta"])['beta'].values.min() / t)
    n_draws = samples.posterior.sizes["chain"] * samples.posterior.sizes["draw"]   # 1200
    ess_min = az.ess(samples, var_names=["beta"])["beta"].values.min()
    print(ess_min, ess_min / n_draws)
    return


@app.cell(disabled=True)
def _(model, pytensor):
    # Profile timing of one execution
    with pytensor.config.change_flags(profile=True, profiling__time_thunks=1):
        model.profile(model.dlogp()).summary()
    return


@app.cell
def _(az, samples):
    az.plot_trace_dist(samples, var_names=["beta"], compact=False)
    return


@app.cell
def _(az, samples):
    az.plot_forest(
        samples,
        var_names=["beta"],
        combined=True,
        ci_probs=[0.5, 0.95],
        figure_kwargs={"figsize": (10, 3)},
    )
    return


@app.cell
def _(HAPLOTYPES, gene_class_counts, lp, np, pl, samples):
    q_hat = np.mean(samples['posterior']['q'], axis=(0,1))
    #q_hat = [np.mean(samples['posterior'][f'q_{i}'], axis=(0,1)) for i in range(n_haps)]
    _dat = pl.concat([pl.DataFrame({
        "haplotype": hap,
        "class": range(len(q_hat[i])),
        "q_hat": q_hat[i].to_numpy(),
        "is_leak": ~gene_class_counts.compat[:,i],
    }) for i, hap in enumerate(HAPLOTYPES)])
    lp.ggplot(_dat, lp.aes(x="class", y="q_hat", color="is_leak")) + lp.facet_wrap("haplotype") + lp.geom_point(tooltips=lp.layer_tooltips(["haplotype", "class"]))
    return


@app.cell
def _(HAPLOTYPES, gene_class_counts, lp, np, pl, samples):
    #weighted_compats = (gene_class_counts.compat / gene_class_counts.compat.sum(axis=1)[:,None])
    class_summaries = (samples['posterior']['q'].values @ gene_class_counts.compat).mean(axis=(0,1))
    #is_leak = gene_class_counts.compat[None, :, :] & np.eye(len(HAPLOTYPES), dtype=bool)[:,None,:]
    #leak_compats = gene_class_counts.compat & gene_class_counts.compat[np.arange(len(HAPLOTYPES))]
    leak_outs = np.array([
        (samples['posterior']['q'].values[:,:,i,:] @ ((~gene_class_counts.compat[:,i,None]) & gene_class_counts.compat)).mean(axis=(0,1))
        for i in range(len(HAPLOTYPES))
    ])
    max_leak = leak_outs.max()
    class_summaries = pl.concat([
        pl.DataFrame({
            "source_hap": hap,
            "out_class": class_summaries[i],
            "leak_to": leak_outs[i],
            "out_hap": HAPLOTYPES,
        })
        for i, hap in enumerate(HAPLOTYPES)
    ])
    lp.gggrid([
        (
            lp.ggplot(class_summaries, lp.aes(y="source_hap", x="out_hap", fill="out_class"))
            + lp.geom_tile()
            + lp.scale_fill_viridis(option="magma", limits=[0,1])
            + lp.ggtitle("Read compatibility matrix")
        ),
        (
            lp.ggplot(class_summaries, lp.aes(x="out_hap", y="source_hap", fill="leak_to"))
            + lp.geom_tile()
            + lp.scale_fill_viridis(option="magma")
            + lp.ggtitle(f"Leaks - max leak = {max_leak:0.2%}")
        )
    ]) + lp.ggsize(900,500)
    return


@app.cell
def _(az, samples):
    az.plot_energy(samples)
    return


if __name__ == "__main__":
    app.run()
