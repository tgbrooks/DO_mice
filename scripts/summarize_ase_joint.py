"""
Summarize the joint ASE model on simulated data, against two-stage runs.

Inputs are per-chunk JSON outputs; each file's run is labelled by its directory name
(e.g. joint, sigmaq_1, oracle_q). For each run reports, over genes:
  * beta calibration: Q = (beta_hat - beta)' V^+ (beta_hat - beta) with V = cov_beta.
    Calibrated: Q ~ chi2_7, so mean/7 = 1, median/6.35 = 1, 5% above 14.07.
    Reported over all of the run's genes and over the genes every run completed.
  * precision: median over common genes of tr(cov_beta) relative to the first run.
For joint-model runs also reports convergence (R-hat, ESS, divergences, separation of
chain means), runtime, and recovery of sigma_u / sigma_gamma against the truth.

Writes the table to --outfile and per-gene values to <outfile>.genes.parquet.
"""

import argparse
import json
import pathlib

import numpy as np
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", help="per-chunk JSON outputs; first run listed is the reference")
parser.add_argument("--outfile", required=True)
parser.add_argument("--sim_dir", default="processed/simulated_classes")
args = parser.parse_args()

CHI2_7_MEDIAN = 6.346
CHI2_7_Q95 = 14.067

truths = pl.read_parquet(f"{args.sim_dir}/gene_truths.parquet")
true_beta = {g: np.asarray(b) for g, b in zip(truths["gene_id"], truths["beta"].to_list())}


def mahalanobis(d, V):
    w, U = np.linalg.eigh(np.asarray(V))
    k = w > w.max() * 1e-8
    z = U[:, k].T @ d
    return float(np.sum(z**2 / w[k]))


rows = []
run_order = []
for path in args.files:
    run = pathlib.Path(path).parent.name.removeprefix("oracle_")
    if run not in run_order:
        run_order.append(run)
    for s in json.load(open(path)):
        g = s["gene_id"]
        if s.get("error") != "none":
            rows.append(dict(run=run, gene_id=g, ok=False, error=s["error"]))
            continue
        V = np.asarray(s["cov_beta"])
        row = dict(
            run=run,
            gene_id=g,
            ok=True,
            error="none",
            Q=mahalanobis(np.asarray(s["beta"]) - true_beta[g], V),
            tr_cov_beta=float(np.trace(V)),
            sigma_u=s["sigma_u"],
            sigma_gamma=s["sigma_gamma"],
        )
        dg = s.get("diagnostics", {})
        if "beta" in dg and isinstance(dg["beta"], dict):  # joint model
            sd = np.sqrt(np.clip(np.diag(V), 1e-12, None))
            chain_means = np.asarray(s["beta_chain_means"])  # chains x haplotypes
            row.update(
                joint=True,
                max_rhat_beta=dg["beta"]["max_rhat"],
                min_ess_bulk_beta=dg["beta"]["min_ess_bulk"],
                max_rhat_sigma_u=dg["sigma_u"]["max_rhat"],
                max_rhat_sigma_gamma=dg["sigma_gamma"]["max_rhat"],
                min_ess_bulk_sigma_gamma=dg["sigma_gamma"]["min_ess_bulk"],
                max_rhat_q=dg["q"]["max_rhat"],
                n_divergences=dg["n_divergences"],
                # largest spread of chain means, in posterior sds
                chain_spread=float(((chain_means.max(0) - chain_means.min(0)) / sd).max()),
                runtime=s.get("runtime"),
            )
        rows.append(row)

genes = pl.DataFrame(rows, infer_schema_length=None)
if "joint" not in genes.columns:
    genes = genes.with_columns(joint=pl.lit(None, pl.Boolean))
genes = (
    genes.join(
        truths.select("gene_id", true_sigma_u="sigma_u", true_sigma_gamma="sigma_gamma"),
        "gene_id",
        how="left",
    )
)
ok = genes.filter("ok")
common = (
    ok.group_by("gene_id")
    .agg(n=pl.col("run").n_unique())
    .filter(pl.col("n") == len(run_order))["gene_id"]
)
ok = ok.with_columns(common=pl.col("gene_id").is_in(common.implode()))
ref_tr = ok.filter(pl.col("run") == run_order[0], "common").select(
    "gene_id", ref_tr="tr_cov_beta"
)


def calibration(df, suffix):
    return [
        (pl.col("Q").mean() / 7).alias(f"Q_mean_ratio{suffix}"),
        (pl.col("Q").median() / CHI2_7_MEDIAN).alias(f"Q_median_ratio{suffix}"),
        (pl.col("Q") > CHI2_7_Q95).mean().alias(f"frac_Q_gt_q95{suffix}"),
    ]


summary = (
    genes.group_by("run")
    .agg(n_genes_ok=pl.col("ok").sum(), n_errors=(~pl.col("ok")).sum())
    .join(ok.group_by("run").agg(*calibration(ok, "")), "run")
    .join(
        ok.filter("common").group_by("run").agg(
            pl.len().alias("n_common"), *calibration(ok, "_common")
        ),
        "run",
    )
    .join(
        ok.filter("common")
        .join(ref_tr, "gene_id")
        .group_by("run")
        .agg(median_tr_cov_vs_first=(pl.col("tr_cov_beta") / pl.col("ref_tr")).median()),
        "run",
    )
    .join(
        ok.group_by("run").agg(
            corr_sigma_u_truth=pl.corr("sigma_u", "true_sigma_u", method="spearman"),
            corr_sigma_gamma_truth=pl.corr("sigma_gamma", "true_sigma_gamma", method="spearman"),
        ),
        "run",
    )
)
jt = ok.filter(pl.col("joint").fill_null(False))
if jt.height:
    converged = (
        (pl.col("max_rhat_beta") < 1.01)
        & (pl.col("n_divergences") == 0)
        & (pl.col("chain_spread") < 1)
    )
    summary = summary.join(
        jt.group_by("run").agg(
            frac_rhat_beta_gt_1_01=(pl.col("max_rhat_beta") > 1.01).mean(),
            frac_rhat_beta_gt_1_05=(pl.col("max_rhat_beta") > 1.05).mean(),
            frac_any_divergence=(pl.col("n_divergences") > 0).mean(),
            frac_chain_spread_gt_1sd=(pl.col("chain_spread") > 1).mean(),
            median_min_ess_bulk_beta=pl.col("min_ess_bulk_beta").median(),
            median_min_ess_bulk_sigma_gamma=pl.col("min_ess_bulk_sigma_gamma").median(),
            frac_rhat_sigma_gamma_gt_1_01=(pl.col("max_rhat_sigma_gamma") > 1.01).mean(),
            frac_rhat_q_gt_1_01=(pl.col("max_rhat_q") > 1.01).mean(),
            median_runtime_s=pl.col("runtime").median(),
            frac_converged=converged.mean(),
            Q_mean_ratio_converged=(pl.col("Q").filter(converged).mean() / 7),
            Q_median_ratio_converged=(pl.col("Q").filter(converged).median() / CHI2_7_MEDIAN),
        ),
        "run",
        how="left",
    )
summary = summary.sort(pl.col("run").replace_strict({r: i for i, r in enumerate(run_order)}))

out = pathlib.Path(args.outfile)
genes.write_parquet(out.with_suffix(".genes.parquet"))
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3, fmt_str_lengths=60):
    text = (
        "ASE beta: joint model vs two-stage on simulated data\n"
        "(calibrated: Q ratios 1, frac_Q_gt_q95 0.05; *_common = genes every run completed;\n"
        f" median_tr_cov_vs_first = precision relative to '{run_order[0]}', < 1 is more precise)\n\n"
        f"{summary.transpose(include_header=True, column_names='run')}\n"
    )
print(text)
out.write_text(text)
