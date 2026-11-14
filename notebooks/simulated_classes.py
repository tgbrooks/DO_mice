import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import yaml
    import marimo as mo
    import polars as pl
    import polars_bio as pb
    import lets_plot as lp
    import numpy as np
    import scipy.stats


    from util.compressed_emase import load_compressed_emase
    from util.compat_classes import get_gene_class_counts, get_gene_totals
    from ase_buffering.ase_model_2steps import prepare_gene_data

    return (
        get_gene_class_counts,
        load_compressed_emase,
        lp,
        mo,
        np,
        pb,
        pl,
        prepare_gene_data,
        scipy,
        yaml,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load data
    """)
    return


@app.cell
def _():
    TISSUE = "simulated_classes"
    return (TISSUE,)


@app.cell
def _():
    SIM_REF_TISSUE = "Adipose"
    return (SIM_REF_TISSUE,)


@app.cell
def _(lp, yaml):
    config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
    HAPLOTYPES = config["haplotypes"].split(",")
    HAPLOTYPE_COLORS = lp.scale_color_brewer(palette="Dark2").palette( len(HAPLOTYPES) )
    return HAPLOTYPES, HAPLOTYPE_COLORS, config


@app.cell
def _(HAPLOTYPES):
    HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}
    return (HAP_TO_HAPNUM,)


@app.cell
def _(TISSUE, config):
    OUTLIER_MOUSE_IDS = config['outlier_ids'][TISSUE]
    return (OUTLIER_MOUSE_IDS,)


@app.cell
def _(TISSUE, pl):
    allele_unique = pl.read_parquet(f"processed/{TISSUE}/allele_unique_reads.parquet")
    return (allele_unique,)


@app.cell
def _(SIM_REF_TISSUE, TISSUE, pl):
    size_factors = pl.read_csv(f"processed/{TISSUE}/size_factors.txt", separator="\t")
    genotypes = pl.read_parquet("processed/genotypes.parquet").select(
        "gene_id",
        "mouse_id",
        "genotype_confidence",
        diplotype="genotype",
    )
    effective_lengths = pl.read_parquet(
        f"processed/{SIM_REF_TISSUE}/effective_lengths.by_gene.parquet"
    )
    return effective_lengths, genotypes, size_factors


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
    def extract_field(field_str):
        entries = field_str.split(".")
        temp = pl.col(entries[0])
        for entry in entries[1:]:
            temp = temp.struct.field(entry)
        return temp

    return (extract_field,)


@app.cell
def _(pl):
    buffering_all2 = pl.read_parquet("results/simulated_classes/buffering_full_results.parquet")
    return (buffering_all2,)


@app.cell
def _(allele_unique, buffering_all2, extract_field, pl, sample_truth):
    _mouse_ids = sorted(allele_unique['mouse_id'].unique())
    etas =  buffering_all2.select(
        "gene_id",
        eta = extract_field("ase_model.eta").list.eval(pl.element().struct.field("mean")),
        eta_se = extract_field("ase_model.eta").list.eval(pl.element().struct.field("sd")),
        mouse_id = extract_field("ase_model.eta").list.eval(pl.element().struct.field("mouse_id")),
    ).explode(
        ["eta", "mouse_id", "eta_se"], empty_as_null=False
    ).join(sample_truth.select('gene_id', 'mouse_id', eta_true='eta'), ['gene_id', 'mouse_id'])
    return (etas,)


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
            **{f"effect_{hap}_se": extract_field(f"ase_model.cov.cov_beta").list.get(i).list.get(i).sqrt()
               for i, hap in enumerate(HAPLOTYPES)},
            **{f"total_{hap}": extract_field(f"total_model.beta.beta[{hap}].mean")
               for i, hap in enumerate(HAPLOTYPES)},
            **{f"total_{hap}_se": extract_field(f"total_model.beta.beta[{hap}].sd")
               for i, hap in enumerate(HAPLOTYPES)},
            sigma_u = extract_field("ase_model.stage2_info.sigma_u"),
            sigma_gamma = extract_field("ase_model.stage2_info.sigma_gamma"),
            ase_median_rhat_eta = extract_field("ase_model.pymc_info.median_rhat_eta"),
            ase_min_ess_bulk_eta = extract_field("ase_model.pymc_info.min_ess_bulk_eta"),
            ase_min_ess_tail_eta = extract_field("ase_model.pymc_info.min_ess_tail_eta"),
            ase_n_diverging_chains = extract_field("ase_model.pymc_info.n_diverging_chains"),
            ase_stage2_converged = extract_field("ase_model.stage2_info.converged"),
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
            total_log_disp = extract_field("total_model.core.log_disp.mean"),
            total_log_disp_se = extract_field("total_model.core.log_disp.sd"),
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
def _(SIM_REF_TISSUE, extract_field, pl):
    ref_buffering2 = (
        pl.read_parquet(f"results/{SIM_REF_TISSUE}/buffering_full_results.parquet")
        .select(
            "gene_id",
            ase_beta = extract_field("ase_model.beta"),
            #ase_beta_se = extract_field("ase_model.beta_se"),
            total_log_disp = extract_field("total_model.core.log_disp.mean"),
            total_log_disp_se = extract_field("total_model.core.log_disp.sd"),
            alpha = extract_field("gene_data.alpha"),
        )
    )
    return (ref_buffering2,)


@app.cell
def _(buffering_all2, extract_field):
    _ase_beta_covs = buffering_all2.select('gene_id', ase_cov = extract_field('ase_model.cov.cov_beta')).rows_by_key("gene_id", unique=True, named=True)
    _total_beta_covs = buffering_all2.select('gene_id', total_cov = extract_field('total_model.beta_cov')).rows_by_key("gene_id", unique=True, named=True)
    beta_covs = {g: {**_ase_beta_covs[g], **_total_beta_covs[g]} for g in _ase_beta_covs.keys()}
    return (beta_covs,)


@app.cell
def _(pl):
    gene_truth = pl.read_parquet("processed/simulated_classes/gene_truths.parquet")
    sample_truth = pl.read_parquet("processed/simulated_classes/sample_truths.parquet")
    return gene_truth, sample_truth


@app.cell
def _(lp):
    # For plots:
    scale_gene_type = lp.scale_color_manual(values={"no_ASE": "grey", "unbuffered": "black", "buffered": "red"})
    gene_type_order = ["no_ASE", "unbuffered", "buffered"]
    return (scale_gene_type,)


@app.cell(disabled=True)
def _(
    HAPLOTYPES,
    OUTLIER_MOUSE_IDS,
    TISSUE,
    allele_unique,
    load_compressed_emase,
):
    # Load ALL the read compatibility data form all the mice
    all_compatibility_classes = {}
    for mouse_id in allele_unique['mouse_id'].unique():
        if mouse_id in OUTLIER_MOUSE_IDS:
            continue
        all_compatibility_classes[mouse_id] = load_compressed_emase(
            f"processed/{TISSUE}/gbrs/{mouse_id}.compressed.h5",
            [f"h{i}" for i in range(len(HAPLOTYPES))],
        )
    return (all_compatibility_classes,)


@app.cell(disabled=True)
def _(
    all_compatibility_classes,
    effective_lengths,
    genotypes,
    get_gene_class_counts,
    prepare_gene_data,
    results,
    tx_annot,
):
    # Gene data objects:
    def _():
        import tqdm
        gene_data = {}
        for gene_id in tqdm.tqdm(results['gene_id']):
            gene_class_counts = get_gene_class_counts(
                gene_id,
                tx_annot,
                all_compatibility_classes,
            )
            gene_effective_lengths = effective_lengths.filter(gene_id=gene_id).select(
                "mouse_id", "haplotype", "effective_length"
            )

            gene_data[gene_id] = prepare_gene_data(
                gene_id,
                gene_class_counts,
                genotypes,
                gene_effective_lengths,
                config=dict(
                    min_genotype_confidence = 0.9,
                )
            )
        return gene_data
    gene_datas = _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assess model performance
    """)
    return


@app.cell
def _(HAPLOTYPES, buffering2, gene_truth, pl, ref_buffering2):
    results = (
        buffering2
            .filter("ase_success", "total_success")
            .with_columns(
                ase_beta = pl.list(pl.col(f"effect_{h}") for h in HAPLOTYPES),
                ase_beta_se = pl.list(pl.col(f"effect_{h}_se") for h in HAPLOTYPES),
                total_beta = pl.list(pl.col(f"total_{h}") for h in HAPLOTYPES),
                total_beta_se = pl.list(pl.col(f"total_{h}_se") for h in HAPLOTYPES),
            )
            .join(gene_truth, "gene_id", suffix="_true")
            .join(ref_buffering2, "gene_id", how="left", suffix="_ref")
            .with_columns(
                gene_type = pl.when(pl.col("has_ase") == False)
                    .then(pl.lit("1. no_ASE"))
                    .when(pl.col("has_buffering"))
                    .then(pl.lit("3. buffered"))
                    .otherwise(pl.lit("2. unbuffered"))
                    .cast(pl.Enum(["1. no_ASE", "2. unbuffered", "3. buffered"])),
            ).with_columns(
                gene_type_order = pl.col("gene_type").to_physical(),
            )
    )
    return (results,)


@app.cell
def _(lp, results):
    lp.gggrid([
        (
            lp.ggplot(results, lp.aes("sigma_u_true", "sigma_u", color="has_ase"))
            + lp.geom_point(alpha=0.3)
            + lp.geom_abline(slope=1,intercept=0)
            + lp.labs(x="sigma_u true", y="sigma_u estimated")
        ),
        (
            lp.ggplot(results, lp.aes("sigma_gamma_true", "sigma_gamma", color="has_ase"))
            + lp.geom_point(alpha=0.3)
            + lp.geom_abline(slope=1,intercept=0)
            + lp.labs(x="sigma_gamma true", y="sigma_gamma estimated")
        )
    ], guides="collect") + lp.ggsize(800,400)
    return


@app.cell
def _(lp, mo, results):
    # In our simulations, phi is the (not log scaled) additional variation needed ontop
    # of the NB variation of each class to achieve the observed reference dispersion
    mo.vstack([
        """In our simulations, variance comes from two main places: negative binomial expression of each read class, and then
        an additional layer of variation affecting all classes simultaneously. This second is because the variation of
        just the read classes does sum up to give the appropriate variance needed in the total counts. This reflects the
        fact that the ASE model has the advantage of being able to ignore substantial inter-mouse variation by working on
        values that are intra-mouse.

        We estimate how much of this second type of variation (encoded in the parameter "phi") to include and here we
        check that it creates data that matches our expectations.
        """,
        lp.gggrid([
            (
                lp.ggplot(results, lp.aes("total_log_disp_ref", "total_log_disp"))
                + lp.geom_point(alpha=0.3)
                + lp.geom_abline(slope=1,intercept=0)
                + lp.labs(x="reference log dispersion", y="estimated log dispersion")
                + lp.geom_smooth(method="loess")
            ),
            (
                lp.ggplot(results, lp.aes("phi", "total_log_disp"))
                + lp.geom_point(alpha=0.3)
                + lp.geom_abline(slope=1,intercept=0)
                + lp.labs(x="phi true", y="estimated log dispersion")
            )
        ]) + lp.ggsize(800, 400)
    ])
    return


@app.cell
def _(HAPLOTYPES, lp, pl, results, scale_gene_type):
    beta_comparison = results.select(
        'gene_id',
        'ase_beta',
        'ase_beta_se',
        'total_beta',
        'total_beta_se',
        'beta',
        "has_ase",
        "has_buffering",
        "gene_type",
        "buffering_factor",
        haplotype=pl.list(pl.lit(h) for h in HAPLOTYPES)
    ).explode("ase_beta", "ase_beta_se", "total_beta", "total_beta_se", "beta", "haplotype", empty_as_null=False)
    beta_comparison
    (
        lp.ggplot(beta_comparison, lp.aes("beta", "ase_beta", color="gene_type"))
        + lp.facet_grid(x="gene_type")
        + lp.geom_point()
        + lp.geom_abline(slope=1, intercept=0, color="black")
        + scale_gene_type
    )
    return (beta_comparison,)


@app.cell
def _(beta_comparison, lp, scale_gene_type):
    (
        lp.ggplot(beta_comparison, lp.aes("beta", "total_beta", color="gene_type"))
        + lp.facet_grid(x="gene_type")
        + lp.geom_point()
        + lp.geom_abline(slope=1, intercept=0, color="black")
        + scale_gene_type
    )
    return


@app.cell
def _(lp, results, scale_gene_type):
    (
        lp.ggplot(results, lp.aes("buffering_factor_true", "buffering_factor", color="gene_type"))
        + lp.facet_grid(x="has_ase")
        + lp.geom_point()
        + lp.geom_abline(slope=1, intercept=0, color="black")
        + scale_gene_type
    )
    return


@app.cell
def _(lp, pl, results, scipy):
    valid_results = (
        results
            .filter(
                pl.col("ase_success"),
                pl.col("total_success"),
                pl.col("sigma_u") < 0.25,
                pl.col("sigma_gamma") < 0.25,
            )
    )
    fdr = (
        valid_results
            .with_columns(fdr = scipy.stats.false_discovery_control(valid_results['buffering_factor_p']))
            .sort("buffering_factor_p")
            .with_columns(
                cum_fp = pl.col("has_buffering").not_().cum_sum(),
            ).with_columns(
                fdp = pl.col("cum_fp") / (pl.row_index() + 1),
            )
    )

    null_ps = (
        valid_results
            .filter(
                has_buffering=False,
            )
            .sort("buffering_factor_p")
            .with_columns(
                expected_p = (pl.row_index() + 0.5)/(pl.len()+1),
            )
    )
    lp.gggrid([
        (
            lp.ggplot(fdr, lp.aes("fdr", "fdp"))
                + lp.geom_line()
                + lp.scale_y_continuous(limits=(0,1))
                + lp.labs(x="FDR (reported)", y="FDP (truth)")
                #+ lp.scale_x_log10()
                #+ lp.scale_y_log10()
                + lp.geom_abline(slope=1, intercept=0, linetype="dashed")
        ),
        (
            lp.ggplot(null_ps, lp.aes("expected_p", "buffering_factor_p"))
                + lp.geom_point(tooltips=lp.layer_tooltips(["gene_id"]))
                + lp.scale_x_log10()
                + lp.scale_y_log10()
                + lp.geom_abline(slope=1, intercept=1, linetype="dashed")
                + lp.labs(x="p-value (theoretical)", y="p-value (reported)")
        )
    ]) + lp.ggsize(800,500)
    return


@app.cell
def _(beta_covs, np, results):
    # Check whether the covariance matrices of the betas are valid
    def calibration(beta_hat, V, beta_true):
        """Batched Mahalanobis check. beta_hat, beta_true: (g, p); V: (g, p, p)."""
        d = beta_hat - beta_true
        w, U = np.linalg.eigh(V)                                  # columns of U are eigenvectors
        k = w > w.max(axis=1, keepdims=True) * 1e-8               # per-gene rank threshold
        z = np.einsum("gij,gi->gj", U, d)                         # U^T d
        Q = np.sum(np.where(k, z**2 / np.where(k, w, 1.0), 0.0), axis=1)
        return Q, k.sum(axis=1)

    null = results.filter(has_buffering=False, has_ase=True)
    rows = null.select("gene_id", "ase_beta", "total_beta", "beta").rows_by_key("gene_id", unique=True, named=True)
    gids = list(rows.keys())
    ase_betas = np.array([rows[g]["ase_beta"] for g in gids])
    total_betas = np.array([rows[g]["total_beta"] for g in gids])
    true_betas = np.array([rows[g]["beta"] for g in gids])        # lambda = 1 here
    covs_ase = np.array([beta_covs[g]["ase_cov"] for g in gids])
    covs_total = np.array([beta_covs[g]["total_cov"] for g in gids])

    Q_ase, dof_ase = calibration(ase_betas, covs_ase, true_betas)
    Q_total, dof_total = calibration(total_betas, covs_total, true_betas)
    return Q_ase, Q_total, dof_ase, dof_total, gids


@app.cell
def _(
    HAPLOTYPES,
    Q_ase,
    Q_total,
    dof_ase,
    dof_total,
    gids,
    lp,
    np,
    pl,
    results,
    scipy,
):
    max_Q = 100
    _t = np.geomspace(1, max_Q, 201)
    _expected = pl.concat([
        pl.DataFrame({
            "Q": _t,
            "density": scipy.stats.chi2.pdf(_t, df=len(HAPLOTYPES)-1),
        }),
    ])

    _Q = pl.concat([
        pl.DataFrame({
            "gene_id": gids,
            "Q": Q_ase,
            "dof": dof_ase,
            "type": "ASE",
        }),
        pl.DataFrame({
            "gene_id": gids,
            "Q": Q_total,
            "dof": dof_total,
            "type": "total",
        })
    ]).join(
        results.select("gene_id", "sigma_u", "sigma_u_true", "sigma_gamma", "sigma_gamma_true", "ase_success", "total_success"),
        "gene_id",
    ).filter(
        "ase_success",
        "total_success",
        #pl.col("sigma_u") < 0.25,
        #pl.col("sigma_gamma") < 0.25,
        #pl.col('sigma_u_true') > 0,
        #pl.col('sigma_u') > 0,
        pl.col('sigma_gamma_true') > 0.08,
        pl.col("sigma_gamma") > 0.08,
    )
    (
        lp.ggplot(_Q, lp.aes(x="Q"))
            + lp.geom_density()
            + lp.geom_line(lp.aes(x="Q", y="density"), data=_expected, color="red") # expected distribution
            + lp.facet_grid(x="type")
            + lp.scale_x_continuous(limits=(0,max_Q))
            #+ lp.scale_y_log10()
            #+ lp.scale_x_log10()
    )
    return


@app.cell
def _(buffering_all2, extract_field, lp, mo, pl, results, scipy):
    lrt = buffering_all2.select(
        'gene_id',
        Q_hat = extract_field("buffering.Q"),
        Q_1 = extract_field("buffering.chi2"),
        chi2_p = extract_field('buffering.chi2_p'),
    ).with_columns(
        lr = pl.col("Q_1") - pl.col("Q_hat"),
    ).join(
        results.select('gene_id', 'has_buffering', 'ase_success', 'total_success', 'sigma_u', 'sigma_gamma'),
        'gene_id',
    )
    lrt = lrt.with_columns(
        # Compute LRT p-values
        lrt_p =  scipy.stats.chi2.sf(lrt['lr'].to_numpy(), df=1)
    ).join(results.filter(
        pl.col("sigma_u") < 0.25,
        pl.col('sigma_gamma') < 0.25,
        ase_success=True,
        total_success=True,
    ).select('gene_id'), how="right", on="gene_id")
    _plt = lp.ggplot(lrt, lp.aes("chi2_p", "lrt_p")) + lp.geom_point(tooltips=lp.layer_tooltips(['gene_id'])) + lp.scale_x_log10() + lp.scale_y_log10() + lp.geom_abline(slope=1, intercept=1)

    _null_ps = (
        lrt
            .filter(
                pl.col("sigma_u") < 0.25,
                pl.col('sigma_gamma') < 0.25,
                has_buffering=False,
                ase_success=True,
                total_success=True,
            )
            .sort("lrt_p")
            .with_columns(
                expected_p = (pl.row_index() + 0.5)/(pl.len()+1),
            )
    )
    _qq_plt = (
        lp.ggplot(_null_ps, lp.aes("expected_p", "lrt_p"))
            + lp.geom_point(tooltips=lp.layer_tooltips(['gene_id']))
            + lp.scale_x_log10()
            + lp.scale_y_log10()
            + lp.geom_abline(slope=1, intercept=1)
    )

    mo.vstack([
        "Compare LRT p-values to Chi2 test p-values",
        lp.gggrid([
            _plt,
            _qq_plt,
        ]) + lp.ggsize(800,300)
    ])
    return


@app.cell
def _(buffering_all2, extract_field, lp, pl, ref_buffering2, results):
    # Compare alphas
    alphas = buffering_all2.select(
        'gene_id',
        est_alpha = extract_field('gene_data.alpha'),
        q = extract_field('gene_data.q_hat'),
    ).join(
        ref_buffering2.select(
            'gene_id',
            true_alpha = "alpha"),
        'gene_id',
        how='left'
    ).join(
        results.filter(
            pl.col("sigma_u") < 0.25,
            pl.col('sigma_gamma') < 0.25,
            ase_success=True,
            total_success=True,
        ).select('gene_id'),
        how="right",
        on="gene_id",
    ).explode(
        ["est_alpha", "true_alpha"],
        empty_as_null=True,
    ).explode(
        ["est_alpha", "true_alpha"],
        empty_as_null=True,
    ).with_columns(
        ratio = pl.col("est_alpha") / pl.col("true_alpha")
    )

    lp.gggrid([
        (
            lp.ggplot(alphas.sample(5_000), lp.aes("true_alpha", "est_alpha"))
                 + lp.geom_pointdensity(show_legend=False)
                + lp.scale_color_viridis(option="magma")
                + lp.geom_abline(slope=1, intercept=1)
                + lp.scale_x_log10()
                + lp.scale_y_log10()
        ),
        (
            lp.ggplot(alphas.sample(5_000), lp.aes("ratio"))
                + lp.geom_density()
                + lp.geom_vline(xintercept=1)
                + lp.scale_x_log10()
                + lp.labs(x="est. alpha / true alpha")
        )
    ]) + lp.ggsize(800,400)
    return


@app.cell
def _(
    HAP_TO_HAPNUM,
    buffering_all2,
    etas,
    extract_field,
    genotypes,
    lp,
    np,
    pl,
    ref_buffering2,
    sample_truth,
):
    # Compute information using true alphas
    # This assess the impact of alpha estimation error on estimated covariance
    def _():
        data = buffering_all2.select(
            'gene_id',
            est_alpha = extract_field('gene_data.alpha'),
            est_q = extract_field('ase_model.q'),
            mouse_id = "mouse_ids_used",
        ).join(
            ref_buffering2.select(
                'gene_id',
                true_alpha = 'alpha',
            ),
            how='left',
            on='gene_id'
        ).explode(
            ["mouse_id", "est_alpha", "true_alpha"],
            empty_as_null=False,
        ).join(
            genotypes,
            how="left",
            on=["gene_id", "mouse_id"],
        ).with_columns(
            hap1 = pl.col("diplotype").str.slice(0,1).replace_strict(HAP_TO_HAPNUM),
            hap2 = pl.col("diplotype").str.slice(1,1).replace_strict(HAP_TO_HAPNUM),
        ).join(
            sample_truth.select('gene_id', 'mouse_id', 'mu', 'p'),
            ['gene_id', 'mouse_id'],
        ).join(
            etas,
            ['gene_id', 'mouse_id'],
        ).group_by(
            "gene_id"
        ).agg(
            "est_alpha", "true_alpha", "hap1", "hap2", "mu", "p", 'eta', pl.col("est_q").first()
        )
        results = []
        for gene_id, dat in data.rows_by_key("gene_id", unique=True, named=True).items():
            alpha_true = np.array(dat['true_alpha'])
            alpha_est = np.array(dat['est_alpha'])
            q = np.array(dat['est_q']) # estimated q
            p = np.array(dat['p']) # note: true p
            hap1 = np.array(dat['hap1'])
            hap2 = np.array(dat['hap2'])
            mu = np.array(dat['mu'])
            total_mean = mu.sum(axis=1)
            def info(mu, alpha):
                return mu / (1 + mu / alpha)
            # per gene: mu (samples x classes, true means), alpha_true, alpha_est, informative-mouse mask
            i_true = info(mu, alpha_true)
            i_est = info(mu, alpha_est)
            # restrict to classes that inform eta: weight by squared sensitivity to p, d mu / d eta
            dmu = total_mean[:, None] * p[:, None] * (1 - p[:, None]) * (q[hap1] - q[hap2])
            w_true = dmu**2 / mu * (i_true / mu)        # Fisher info about eta per class, true alpha
            w_est = dmu**2 / mu * (i_est / mu)          # same, with estimated alpha
            #inflation = (w_est**2 / w_true).sum(axis=1) / w_est.sum(axis=1)   # per mouse
            ratio_terms = np.divide(w_est**2, w_true, out=np.zeros_like(w_est), where=w_true > 0)
            inflation = ratio_terms.sum(axis=1) / w_est.sum(axis=1)
            # only the mice present at this point are the informative ones with estimated 'etas'
            gene_inflation = np.median(inflation)
            results.append({"gene_id": gene_id, "gene_inflation": gene_inflation})
        return pl.DataFrame(results)
    _df = _().join(
        buffering_all2.select('gene_id', extract_field('buffering.Q')),
        'gene_id'
    )
    lp.ggplot(_df, lp.aes("Q", "gene_inflation")) + lp.geom_point()
    return


@app.cell
def _(etas, lp, np, pl, scipy):
    # Check eta standard variances
    _df = etas.select(
        z = (pl.col('eta') - pl.col("eta_true")) / pl.col('eta_se')
    )
    _t = np.linspace(-4,4, 101)
    _expected = pl.DataFrame({
        "z": _t,
        "density": scipy.stats.norm.pdf(_t),
    })
    (
        lp.ggplot(_df, lp.aes("z"))
            + lp.geom_density()
            + lp.geom_line(lp.aes("z", "density"), data=_expected, color="red", linetype="dashed")
            + lp.labs(x="eta z-score ((est - true) / se)")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gene plots - simulated data
    """)
    return


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
                    .sort("total", "mouse_id", "class")
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


@app.cell(hide_code=True)
def _(HAPLOTYPES, buffering2, etas, gene_selector, genotypes, lp, mo, pl):
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
                .with_columns(genotype="diplotype")
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
    _df = etas.filter(gene_id=gene_selector.value).join(genotypes,['gene_id', 'mouse_id']).with_columns(
            eta_lo = pl.col("eta") - 1.96 * pl.col("eta_se"),
            eta_hi = pl.col("eta") + 1.96 * pl.col("eta_se"),
    )
    mo.vstack([
        lp.ggplot(_df, lp.aes(x="eta_true", y="eta"))
            + lp.geom_point(tooltips=lp.layer_tooltips(["mouse_id", "diplotype"]))
            + lp.geom_errorbar(lp.aes(ymin="eta_lo", ymax="eta_hi"))
            + lp.geom_abline(slope=1, intercept=0),
        _()
    ])
    return


if __name__ == "__main__":
    app.run()
