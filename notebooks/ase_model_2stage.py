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
    from ase_buffering.ase_model import make_ase_model, summarize_read_classes, summarize_bias_rates, estimate_class_proportions, estimate_class_alpha
    from ase_buffering.ase_model_2steps import make_stage1_model, make_stage2_model, prepare_gene_data, stage1_likelihood
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
    genotypes = pl.read_parquet("processed/genotypes.parquet").select(
        "gene_id",
        "mouse_id",
        "genotype_confidence",
        diplotype="genotype",
    )
    return (genotypes,)


@app.cell
def _(allele_unique):
    mouse_ids = sorted(allele_unique["mouse_id"].unique())
    return (mouse_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model ASE with bias correction
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
def _(all_compatibility_classes, effective_lengths, gene_selector, tx_annot):
    gene_class_counts_raw =  get_gene_class_counts(
            gene_selector.value, tx_annot, all_compatibility_classes
    )
    gene_class_counts = gene_class_counts_raw
    gene_effective_lengths = effective_lengths.filter(gene_id=gene_selector.value).select(
        "mouse_id", "haplotype", "effective_length"
    )
    return gene_class_counts, gene_effective_lengths


@app.cell
def _(gene_class_counts, gene_effective_lengths, gene_selector, genotypes):
    gene_data = prepare_gene_data(
        gene_selector.value,
        gene_class_counts,
        genotypes,
        gene_effective_lengths,
        config=dict(
            min_genotype_confidence = 0.9,
        )
    )
    return (gene_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ASE Model Stage 1
    """)
    return


@app.cell
def _(gene_data):
    stage1_config = dict(
        q_mode="adaptive_centered",  # or "q_hat"
        sigma_q_sd=0.5,
        min_counts_to_center=50,
        eta_prior_sd=3.0,  # weak independent prior on each mouse's eta (divided out later)
        min_eta_info=1.0,  # mice with less plug-in information are not fit for eta
    )
    ase_model1 = make_stage1_model(
        gene_data,
        stage1_config,
    )
    return ase_model1, stage1_config


@app.cell
def _(HAPLOTYPES, HAP_TO_HAPNUM, gene_class_counts, gene_selector, genotypes):
    # For plotting
    n_samples = int(gene_class_counts.counts.shape[0])
    n_classes = int(gene_class_counts.compat.shape[0])
    n_haps = len(HAPLOTYPES)
    gene_diplotypes = (
        genotypes.filter(gene_id=gene_selector.value)
        .select("mouse_id", "diplotype", "genotype_confidence")
        .join(
            pl.DataFrame({"mouse_id": gene_class_counts.ids}),
            "mouse_id",
            maintain_order="right",
        ).with_columns(
            hap1=pl.col("diplotype").str.slice(0, 1).replace_strict(HAP_TO_HAPNUM),
            hap2=pl.col("diplotype").str.slice(1, 1).replace_strict(HAP_TO_HAPNUM),
        )
    )
    hap1 = gene_diplotypes["hap1"].to_numpy()
    hap2 = gene_diplotypes["hap2"].to_numpy()
    return hap1, hap2, n_classes


@app.cell
def _(ase_model1, mo):
    mo.mermaid(pm.model_to_mermaid(ase_model1))
    return


@app.cell
def _(ase_model1):
    # Fit the model
    _start = time.time()
    with ase_model1:
        idata_ase1 = pm.sample(
            300,
            chains=6,
            cores=6,
            blas_cores=1,
            random_seed=100,
            nuts_sampler="nutpie",
        )
    _end = time.time()
    ase_model_time1 = _end - _start
    return (idata_ase1,)


app._unparsable_cell(
    r"""
    def _():
        eta = idata_ase1.posterior['eta'].mean(axis=(1)).data
        sample_ids = idata_ase1.posterior.coords['fit_samples'].data
        diplo = gene_diplotypes.join(pl.DataFrame({"mouse_id": sample_ids}), "mouse_id", how="right", maintain_order="right")
        diplotype = np.broadcast_to(diplo['diplotype'], eta.shape)
        chain = np.broadcast_to(np.arange(eta.shape[0])[:,None], eta.shape).astype(str)
        mouse_id = np.broadcast_to(diplo['mouse_id'], eta.shape)
        info = np.broadcast_to(gene_data.info[idata_ase1., eta.shape) #TODO: take the idata_as1.posterior.coords.fit_samples.data mice from gene_data.info
        return lp.ggplot(pl.DataFrame({'mouse_id': mouse_id.flatten(), 'diplotype': diplotype.flatten(), 'eta': eta.flatten(), 'chain': chain.flatten(), "eta_info": info.flatten()}).sample(fraction=1.0).sort('diplotype'), lp.aes('diplotype', 'eta', color="info")) + lp.geom_point(alpha=0.5, tooltips=lp.layer_tooltips(["mouse_id", "diplotype", "chain"]), position=lp.position_dodge(width=.8))
    mo.vstack([
        "Per-mouse imbalance estimates (eta) by diplotype + chain",
        _()
    ])
    """,
    name="_"
)


@app.cell
def _(idata_ase1):
    az.summary(idata_ase1, var_names="eta", filter_vars="like")
    return


@app.cell
def _(idata_ase1):
    az.summary(idata_ase1, var_names="sigma", filter_vars="like")
    return


@app.cell(hide_code=True)
def _(HAPLOTYPES, gene_class_counts, idata_ase1, lp, n_classes):
    q = np.mean(idata_ase1["posterior"]["q"], axis=(0, 1))
    q_se = np.std(idata_ase1["posterior"]["q"], axis=(0, 1))
    # q = [np.mean(samples['posterior'][f'q_{i}'], axis=(0,1)) for i in range(n_haps)]
    _dat = pl.concat(
        [
            pl.DataFrame(
                {
                    "haplotype": hap,
                    "class": range(len(q[i])),
                    "q": q[i].to_numpy(),
                    "q_se": q_se[i].to_numpy(),
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
        y=pl.col("q"),
        ymin=pl.col("q") - 1.96 * pl.col("q_se"),
        ymax=pl.col("q") + 1.96 * pl.col("q_se"),
    )
    (
        lp.ggplot(_dat, lp.aes(x="class", y="y", color="is_leak"))
        + lp.facet_wrap("haplotype")
        + lp.geom_point(
            tooltips=lp.layer_tooltips(
                ["haplotype", "class", "q", "ymin", "ymax", "compat_with"]
            )
        )
        + lp.geom_errorbar(lp.aes(ymin="ymin", ymax="ymax"))
        + lp.scale_color_manual(breaks=[False, True], values=["black", "red"])
        + lp.ggtb()
        + lp.labs(y="q")
        + lp.scale_y_log10()
        + lp.ggtitle("Class proportions by haplotype")
    )
    return


@app.cell(hide_code=True)
def _(gene_class_counts, idata_ase1, lp):
    class_summaries = summarize_read_classes(idata_ase1, gene_class_counts)
    max_leak = class_summaries['leak_to'].max()
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
def _(HAPLOTYPES, HAPLOTYPE_COLORS, gene_class_counts, idata_ase1, lp):
    # Plot fit allele-unique bias rates
    _df = summarize_bias_rates(idata_ase1, gene_class_counts)
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
def _(idata_ase1):
    az.plot_energy(idata_ase1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ASE Model Stage 2
    """)
    return


@app.cell
def _(gene_data, idata_ase1, stage1_config):
    handoff = stage1_likelihood(
        idata_ase1,
        gene_data,
        config=dict(
            stage2_cov="full", 
            cov_shrink=0.1,  # shrink stage-1 posterior correlations toward 0 (full mode)
            max_abs_skew=0.5,  # flag mice whose stage-1 posterior is clearly non-Gaussian
            eta_prior_sd = stage1_config['eta_prior_sd'],
            min_eta_info = stage1_config['min_eta_info'],
        )
    )
    return (handoff,)


@app.cell
def _(gene_data, handoff):
    ase_model2 = make_stage2_model(
        gene_data,
        handoff,
        config=dict(
            gamma_mode="mvn",  # or "zero"
            beta_sd=0.5,
            sigma_gamma_sd=1.0,
            sigma_u_sd=0.3,
            stage2_cov="full",  # "full" or "diag"
        )
    )
    return (ase_model2,)


@app.cell
def _(ase_model2, mo):
    mo.mermaid(pm.model_to_mermaid(ase_model2))
    return


@app.cell
def _(ase_model2):
    # Fit the model - stage 2
    _start = time.time()
    with ase_model2:
        idata_ase2 = pm.sample(
            300,
            chains=6,
            cores=6,
            blas_cores=1,
            random_seed=100,
            nuts_sampler="nutpie",
        )
    _end = time.time()
    ase_model_time2 = _end - _start
    return (idata_ase2,)


@app.cell
def _(gene_data, idata_ase2, lp, mo):
    def _():
        delta = idata_ase2.posterior['delta'].mean(axis=(1)).data
        hap_pairs = np.broadcast_to(idata_ase2.posterior.coords['hap_pairs'].data, delta.shape)
        chain = np.broadcast_to(np.arange(delta.shape[0])[:,None], delta.shape).astype(str)
        beta = idata_ase2.posterior['beta'].mean(axis=(0,1)).data
        beta_fit = np.broadcast_to((gene_data.Bm @ beta), delta.shape)
        beta_all = idata_ase2.posterior['beta'].mean(axis=(1)).data
        beta_chain = np.broadcast_to(np.arange(beta_all.shape[0])[:,None], beta_all.shape).astype(str)
        haplotypes = np.broadcast_to(idata_ase2.posterior.coords['haplotypes'].data, beta_all.shape)
        return lp.gggrid([
            (
                lp.ggplot(
                    pl.DataFrame({
                        'hap_pair': hap_pairs.flatten(),
                        'delta': delta.flatten(),
                        'beta_fit': beta_fit.flatten(),
                        'chain': chain.flatten()}
                    ),
                    lp.aes('hap_pair', 'delta', color="chain")
                )
                    + lp.geom_point(
                        tooltips=lp.layer_tooltips(["hap_pair", "delta", "chain"]),
                        position=lp.position_dodge(width=.8)
                    )
                    + lp.geom_crossbar(lp.aes(y="beta_fit"), color="black", fatten=1)
            ),
            (
                lp.ggplot(
                    pl.DataFrame({
                        "haplotype": haplotypes.flatten(),
                        "beta": beta_all.flatten(),
                        "chain": beta_chain.flatten(),
                    }),
                    lp.aes(x="haplotype", y="beta", color="chain")
                )
                + lp.geom_point(
                    position=lp.position_dodge(width=0.8)
                )
            )
        ], guides="collect", sharey=True, align=True) + lp.ggsize(1000, 500)
    mo.vstack([
        "Fit delta values for each (heterozygous) pair. Black bars denote the fits using just beta (additive haplotype effects, no diplotype effect).",
        _()
    ])
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


@app.cell(disabled=True, hide_code=True)
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

    #warm up
    wrapped_dlogp(ip)

    n = 100
    t = time.perf_counter()
    for _ in range(n):
        #logp, grad = f(x0)
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
        HAP_TO_HAPNUM = {hap:i for i,hap in enumerate(HAPLOTYPES)}
        hapnums = diplotypes.select(
            hap1=pl.col("diplotype").str.slice(0,1).replace_strict(HAP_TO_HAPNUM),
            hap2=pl.col("diplotype").str.slice(1,1).replace_strict(HAP_TO_HAPNUM),
        )
        return hapnums['hap1'].to_numpy(), hapnums['hap2'].to_numpy()

    return


@app.cell(disabled=True)
def _(HAPLOTYPES, HAP_TO_HAPNUM, allele_unique, get_gene_class_counts_cached):
    # Relationship of variance and class prop in homozygous mice
    _gene_list = [x.strip() for x in """
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
    def _():
        temp =[]
        for gene_id in _gene_list:
            gene_class_counts =  get_gene_class_counts_cached( gene_id )
            gene_diplotypes = (
                allele_unique.filter(gene_id=gene_id)
                .select("diplotype", "mouse_id")
                .join(
                    pl.DataFrame({"mouse_id": gene_class_counts.ids}),
                    "mouse_id",
                    maintain_order="right",
                ).with_columns(
                    hap1=pl.col("diplotype").str.slice(0, 1).replace_strict(HAP_TO_HAPNUM),
                    hap2=pl.col("diplotype").str.slice(1, 1).replace_strict(HAP_TO_HAPNUM),
                )
            )
            q_hat = estimate_class_proportions(gene_class_counts.counts, gene_diplotypes['hap1'].to_numpy(), gene_diplotypes['hap2'].to_numpy(), len(HAPLOTYPES))
            # NOTE: this works best if diplotypes are always sorted (AB not BA), which they are in
            # our dataset but may not be in general (e.g., simulated). Estimate is just noisier otherwise.
            for (diplotype, hap1, hap2), dat in gene_diplotypes.group_by(["diplotype", "hap1", "hap2"]):
                these_mice = dat['mouse_id']
                these_mice_ids = np.where(np.isin(gene_class_counts.ids, these_mice))
                if len(these_mice) < 4:
                    continue
                these_counts=gene_class_counts.counts[these_mice_ids]
                tots = np.sum(these_counts, axis=1)
                q_hat_close = np.abs(np.log(q_hat[hap1] / q_hat[hap2])) < 0.5
                with np.errstate(divide="ignore", invalid="ignore"):
                    props = these_counts / np.sum(these_counts, axis=1, keepdims=True)
                    mprops = np.mean(props, axis=0)
                    mult_var = (props * (1-props) / these_counts.sum(axis=1, keepdims=True)).mean(axis=0)
                    true_var = props.var(axis=0, ddof=1)
                    excess_var = true_var - mult_var
                    alpha = mprops**2/(true_var - mprops*np.mean(1/tots))
                n_classes = gene_class_counts.compat.shape[0]
                temp.append(pl.DataFrame({
                    "gene_id": gene_id,
                    "diplotype": diplotype,
                    "var": true_var,
                    "mean_T": np.mean(tots),
                    "mean_inv_T": np.mean(1/tots),
                    "alpha": alpha,
                    "q_hat_close": q_hat_close,
                    "prop": mprops,
                    "n_samples": len(these_mice),
                    "class_num": np.arange(mprops.shape[0]),
                    "hap_str": [''.join(h for i,h in enumerate(HAPLOTYPES) if gene_class_counts.compat[j,i]) for j in range(n_classes)]
                }))
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
        #pl.col("alpha") > 0,
    ).with_columns(
        alpha_nice = pl.when(pl.col("alpha") < 0)
            .then(1e3)
            .otherwise("alpha"),
        disp = 1/pl.col("alpha"),
    )
    _X =  np.array([
            np.ones(len(_df)),
            np.log(_df['prop'].to_numpy()),
            np.log(_df['mean_T'].to_numpy()),
        ]).T
    res, _, _, _ = np.linalg.lstsq(
        _X,
        np.log(_df['var'].to_numpy())
    )
    _var = np.exp(res[0] + np.log(_df['prop'])*res[1] + np.log(_df['mean_T'])*res[2])
    _fit = pl.DataFrame({"prop": _df['prop'], 'mean_T': _df['mean_T'], "gene_id":_df['gene_id'], "var": _var})
    print(res)
    (
        lp.ggplot(
            _df.with_columns(
                is_hom = pl.col("diplotype").is_in([f'{hap}{hap}' for hap in HAPLOTYPES]),
                #compatible = pl.col("diplotype").is_in(pl.col("hap_str").implode())
            ).sample(fraction=1.0),# shuffle
            lp.aes("prop", "var", color="is_hom")
        )
        + lp.facet_wrap("gene_id")
        + lp.geom_point(tooltips=lp.layer_tooltips(["class_num", "n_samples", "diplotype", "hap_str"]), alpha=0.3)
        + lp.scale_x_log10()
        + lp.scale_y_continuous(trans="log10")
        + lp.geom_smooth(lp.aes("prop", "var"), inherit_aes=False, method="lm", deg=4)
        #+ lp.geom_smooth(lp.aes("prop", "var"), inherit_aes=False, method="lm")
        + lp.geom_line(data=_fit, color="black")
    )
    return


@app.cell
def _(HAPLOTYPES, gene_class_counts, hap1, hap2):
    _q_hat = estimate_class_proportions(gene_class_counts.counts, hap1, hap2, len(HAPLOTYPES))
    #q_hat
    #_hap1, _hap2
    var_fit, alpha_fit = estimate_class_alpha(gene_class_counts, _q_hat, hap1, hap2)
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
def _(idata_ase1, mo):
    def extra_dims(var):
        """Dims of a variable other than chain/draw."""
        return [d for d in idata_ase1.posterior[var].dims if d not in ("chain", "draw")]

    def coord_pickers(var):
        """One dropdown per extra dim of `var` (empty dict if scalar variable)."""
        da = idata_ase1.posterior[var]
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

    plt_var_names = list(idata_ase1.posterior.data_vars)
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
def _(idata_ase1, lp, x_coords, x_var, y_coords, y_var):
    def _():
        def select(var, coords):
            da = idata_ase1.posterior[var].sel(**coords.value)  # -> (chain, draw)
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
                "chain": [str(c) for c in xda["chain"].values for _ in xda["draw"].values],
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
        x = idata_ase.posterior['q_logit'].mean(axis=(0,1)).data
        hap = np.broadcast_to(idata_ase.posterior['q_logit'].coords['haplotypes'].data[:,None], x.shape).astype(str)
        class_name = np.broadcast_to(idata_ase.posterior['q_logit'].coords['classes'].data[None,:], x.shape).flatten()
        q_hat = idata_ase.constant_data['q_hat'].data
        pinned = idata_ase.constant_data['q_logit_pinned'].data
        return (
            lp.ggplot(pl.DataFrame({'x': x.flatten(), 'y': x.flatten(), 'hap': hap.flatten(), 'q_hat': q_hat.flatten(), "class_name": class_name, "pinned": pinned.flatten()}), lp.aes('q_hat', 'y', color='hap', shape="pinned")) 
            + lp.geom_point(alpha=1.0, tooltips=lp.layer_tooltips(["hap", "class_name"]))
            + lp.scale_x_log10()
            + lp.labs(y="q_logit")
            + lp.ggtb()
        )
        #gene_class_counts.counts[hap1 == 0, 4]
    mo.vstack([
        "Compare q_hat (naive estimate with no ASE) to the adjustments to q_hat (on a logit scale) that the full model makes",
        _()
    ])
    return


if __name__ == "__main__":
    app.run()
