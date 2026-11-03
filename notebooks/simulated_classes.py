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

    return json, lp, mo, np, pb, pl, scipy, yaml


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
def _(lp, yaml):
    config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
    HAPLOTYPES = config["haplotypes"].split(",")
    HAPLOTYPE_COLORS = lp.scale_color_brewer(palette="Dark2").palette( len(HAPLOTYPES) )
    return HAPLOTYPES, config


@app.cell
def _(TISSUE, config):
    OUTLIER_MOUSE_IDS = config['outlier_ids'][TISSUE]
    return


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
def _(buffering_with_bias_objs, pl):
    buffering_all2 = pl.DataFrame(buffering_with_bias_objs)
    return (buffering_all2,)


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
def _(HAPLOTYPES, TISSUE, json, np):
    def load_objects(TISSUE):
        error_genes = []
        out = []
        eta_out = []
        beta_covs_out = {}
        with open(f"results/{TISSUE}/buffering_with_bias.ndjson") as f:
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
                        "log_disp": get("total_model.core.log_disp.mean"),
                        "log_disp_se": get("total_model.core.log_disp.sd"),
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
    buffering_with_bias_objs, eta_objs, beta_covs = load_objects(TISSUE)
    ref_objs, _, _ = load_objects("Adipose")
    return buffering_with_bias_objs, ref_objs


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
            total_log_disp = extract_field("total_model.log_disp"),
            total_log_disp_se = extract_field("total_model.log_disp_se"),
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
def _(extract_field, pl, ref_objs):
    ref_buffering2 = (
        pl.DataFrame(ref_objs)
        .select(
            "gene_id",
            ase_beta = extract_field("ase_model.beta"),
            ase_beta_se = extract_field("ase_model.beta_se"),
            total_log_disp = extract_field("total_model.log_disp"),
            total_log_disp_se = extract_field("total_model.log_disp_se"),
        )
    )
    return (ref_buffering2,)


@app.cell
def _(pl):
    gene_truth = pl.read_parquet("processed/simulated_classes/gene_truths.parquet")
    sample_truth = pl.read_parquet("processed/simulated_classes/sample_truths.parquet")
    return (gene_truth,)


@app.cell
def _(lp):
    # For plots:
    scale_gene_type = lp.scale_color_manual(values={"no_ASE": "grey", "unbuffered": "black", "buffered": "red"})
    gene_type_order = ["no_ASE", "unbuffered", "buffered"]
    return (scale_gene_type,)


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
                total_beta = pl.list(pl.col(f"total_{h}") for h in HAPLOTYPES),
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
    ]) + lp.ggsize(800,400)
    return


@app.cell
def _(lp, results):
    # In our simulations, phi is the (not log scaled) additional variation needed ontop
    # of the NB variation of each class to achieve the observed reference dispersion
    lp.gggrid([
        (
            lp.ggplot(results, lp.aes("total_log_disp_ref", "total_log_disp"))
            + lp.geom_point(alpha=0.3)
            + lp.geom_abline(slope=1,intercept=0)
            + lp.labs(x="reference log dispersion", y="estimated log dispersion")
        ),
        (
            lp.ggplot(results, lp.aes("phi", "total_log_disp"))
            + lp.geom_point(alpha=0.3)
            + lp.geom_abline(slope=1,intercept=0)
            + lp.labs(x="phi true", y="estimated log dispersion")
        )
    ]) + lp.ggsize(800, 400)
    return


@app.cell
def _(HAPLOTYPES, lp, pl, results, scale_gene_type):
    beta_comparison = results.select(
        'gene_id',
        'ase_beta',
        'total_beta',
        'beta',
        "has_ase",
        "has_buffering",
        "gene_type",
        "buffering_factor",
        haplotype=pl.list(pl.lit(h) for h in HAPLOTYPES)
    ).explode("ase_beta", "total_beta", "beta", "haplotype", empty_as_null=False)
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
    fdr = (
        results
            .with_columns(fdr = scipy.stats.false_discovery_control(results['buffering_factor_p']))
            .sort("buffering_factor_p")
            .with_columns(
                cum_fp = pl.col("has_buffering").not_().cum_sum(),
            ).with_columns(
                fdp = pl.col("cum_fp") / (pl.row_index() + 1),
            )
    )
    (
        lp.ggplot(fdr, lp.aes("fdr", "fdp"))
            + lp.geom_line()
            + lp.scale_y_continuous(limits=(0,1))
            + lp.geom_abline(slope=1, intercept=0, linetype="dashed")
            + lp.labs(x="FDR (estimated)", y="FDP (truth)")
            #+ lp.scale_x_log10()
    )
    return


if __name__ == "__main__":
    app.run()
