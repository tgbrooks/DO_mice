"""
Summarize ASE beta calibration of the oracle / sigma_q-sweep runs on simulated data.

Each input file's run is labelled by its directory: oracle_<setting> -> <setting>
(none / alpha / q / both), anything else (e.g. sigmaq_1) as is. For each run reports,
over genes:
  * Q = (beta_hat - beta)' V^+ (beta_hat - beta), with V = cov_beta (sandwich) and,
    for comparison, the previous stage-2 method (eb_comparison). Under correct
    calibration Q ~ chi2_7: mean/7 = 1, median/6.35 = 1, 5% above 14.07.
  * Diplotype-group z-scores of the stage-1 eta errors: the error of the mean eta of
    each diplotype group, divided by the sd stage 1 assigns it assuming independent
    mice. Variance ~1 means no unmodelled error shared within a diplotype.

Also writes a per-gene parquet next to the output, for drilling into.
"""

import argparse
import json
import pathlib

import numpy as np
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("part2", nargs="+", help="oracle_*/...ase_model_part2.json files")
parser.add_argument("--outfile", required=True)
parser.add_argument("--sim_dir", default="processed/simulated_classes")
args = parser.parse_args()

CHI2_7_MEDIAN = 6.346
CHI2_7_Q95 = 14.067

gene_truths = pl.read_parquet(f"{args.sim_dir}/gene_truths.parquet")
true_beta = {
    g: np.asarray(b)
    for g, b in zip(gene_truths["gene_id"], gene_truths["beta"].to_list())
}
sample_truths = pl.read_parquet(f"{args.sim_dir}/sample_truths.parquet").select(
    "gene_id", "mouse_id", eta_true="eta"
)


def mahalanobis(d, V):
    w, U = np.linalg.eigh(np.asarray(V))
    k = w > w.max() * 1e-8
    z = U[:, k].T @ d
    return float(np.sum(z**2 / w[k])), int(k.sum())


gene_rows = []
mouse_rows = []
for path in args.part2:
    oracle = pathlib.Path(path).parent.name.removeprefix("oracle_")
    for s in json.load(open(path)):
        if s.get("error") != "none":
            gene_rows.append(dict(oracle=oracle, gene_id=s["gene_id"], error=s["error"]))
            continue
        g = s["gene_id"]
        d = np.asarray(s["beta"]) - true_beta[g]
        Q, rank = mahalanobis(d, s["cov_beta"])
        row = dict(
            oracle=oracle,
            gene_id=g,
            error="none",
            Q=Q,
            rank=rank,
            beta_norm=float(np.linalg.norm(true_beta[g])),
            sigma_u=s["sigma_u"],
            sigma_gamma=s["sigma_gamma"],
            tr_cov_beta_over_working=s["diagnostics"]["tr_cov_beta_over_working"],
            # stage-1 sampling health (absent if ESS/R-hat could not be computed)
            min_ess_bulk_eta=s["diagnostics"].get("min_ess_bulk_eta"),
            max_rhat_eta=s["diagnostics"].get("max_rhat_eta"),
        )
        eb = s.get("eb_comparison", {})
        if "cov_beta" in eb:
            row["Q_eb"] = mahalanobis(np.asarray(eb["beta"]) - true_beta[g], eb["cov_beta"])[0]
        gene_rows.append(row)
        p = s["per_mouse"]
        mouse_rows.append(
            pl.DataFrame(
                dict(
                    oracle=oracle,
                    gene_id=g,
                    mouse_id=p["fit_mouse_ids"],
                    diplotype=p["diplotype"],
                    eta_hat=p["eta_lik_mean"],
                    eta_var=p["eta_lik_var"],
                    offset=p["log_eff_ratio"],
                )
            )
        )

genes = pl.DataFrame(gene_rows, infer_schema_length=None)
mice = pl.concat(mouse_rows).join(sample_truths, ["gene_id", "mouse_id"])
groups = (
    mice.with_columns(err=pl.col("eta_hat") - pl.col("eta_true"))
    .group_by("oracle", "gene_id", "diplotype")
    .agg(
        n=pl.len(),
        mean_err=pl.col("err").mean(),
        se=pl.col("eta_var").sum().sqrt() / pl.len(),
        true_delta=(pl.col("eta_true") - pl.col("offset")).mean(),
        z_mouse_var=(pl.col("err") / pl.col("eta_var").sqrt()).var(),
    )
    .filter(pl.col("n") >= 5)
    .with_columns(z=pl.col("mean_err") / pl.col("se"))
)

ok = genes.filter(error="none")
summary = (
    ok.group_by("oracle")
    .agg(
        n_genes=pl.len(),
        Q_mean_ratio=pl.col("Q").mean() / 7,
        Q_median_ratio=pl.col("Q").median() / CHI2_7_MEDIAN,
        frac_Q_gt_q95=(pl.col("Q") > CHI2_7_Q95).mean(),
        Q_eb_mean_ratio=pl.col("Q_eb").mean() / 7,
        Q_eb_median_ratio=pl.col("Q_eb").median() / CHI2_7_MEDIAN,
        median_sandwich_over_working=pl.col("tr_cov_beta_over_working").median(),
        median_min_ess_bulk_eta=pl.col("min_ess_bulk_eta").median(),
        frac_rhat_eta_gt_1_01=(pl.col("max_rhat_eta") > 1.01).mean(),
    )
    .join(
        groups.group_by("oracle").agg(
            group_z_var=pl.col("z").var(),
            group_corr_err_delta=pl.corr("mean_err", "true_delta"),
            mouse_z_var=pl.col("z_mouse_var").median(),
        ),
        "oracle",
    )
    .join(
        genes.group_by("oracle").agg(n_errors=(pl.col("error") != "none").sum()),
        "oracle",
    )
    .with_columns(
        _order=pl.col("oracle").replace_strict(
            {"none": 0, "alpha": 1, "q": 2, "both": 3}, default=4, return_dtype=pl.Int64
        ),
        # sigma_q sweep runs (directories sigmaq_<value>) sort numerically
        _sigma_q=pl.col("oracle").str.extract(r"^sigmaq_(.+)$").cast(pl.Float64, strict=False),
    )
    .sort("_order", "_sigma_q")
    .drop("_order", "_sigma_q")
)

out = pathlib.Path(args.outfile)
genes.write_parquet(out.with_suffix(".genes.parquet"))
groups.write_parquet(out.with_suffix(".groups.parquet"))
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3):
    text = (
        "ASE beta calibration under oracle inputs (ideal: ratios 1, frac_Q_gt_q95 0.05,\n"
        "group_z_var 1, group_corr_err_delta 0, mouse_z_var 1)\n\n"
        f"{summary.transpose(include_header=True, column_names='oracle')}\n"
    )
print(text)
out.write_text(text)
