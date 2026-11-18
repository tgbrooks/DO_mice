"""
End-to-end check of the buffering pipeline on the class-level simulation.

Reads the collected buffering results (results/simulated_classes/
buffering_full_results.parquet) and the simulation truth, and reports, by gene type:
  * no_ase:       true beta = 0 (lambda undefined)
  * ase_no_buff:  true lambda = 1
  * ase_buff:     true lambda = buffering_factor < 1
the lambda estimates (bias, error, CI coverage), the lambda = 1 test (fraction of
p < 0.05: should be 0.05 for no_ase / ase_no_buff, power for ase_buff), and the
calibration of the ASE and total-model betas (Q, chi2_7 when calibrated; the total
model's true beta is lambda * beta).

Writes the table to --outfile and per-gene values to <outfile>.genes.parquet.
"""

import argparse
import pathlib

import numpy as np
import polars as pl
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results/simulated_classes/buffering_full_results.parquet")
parser.add_argument("--truths", default="processed/simulated_classes/gene_truths.parquet")
parser.add_argument("--outfile", required=True)
args = parser.parse_args()

HAPLOTYPES = yaml.safe_load(open("config.yaml"))["haplotypes"].split(",")
CHI2_7_MEDIAN = 6.346
CHI2_7_Q95 = 14.067


def mahalanobis(d, V):
    w, U = np.linalg.eigh(np.asarray(V))
    k = w > w.max() * 1e-8
    z = U[:, k].T @ d
    return float(np.sum(z**2 / w[k]))


res = pl.read_parquet(args.results)
truths = pl.read_parquet(args.truths)
truth = {r["gene_id"]: r for r in truths.select(
    "gene_id", "has_ase", "has_buffering", "buffering_factor", "beta"
).to_dicts()}

rows = []
n_errors = 0
for r in res.to_dicts():
    if r.get("error") != "none":
        n_errors += 1
        continue
    t = truth[r["gene_id"]]
    beta = np.asarray(t["beta"])
    lam = float(t["buffering_factor"]) if t["has_buffering"] else 1.0
    tm = r["total_model"]
    beta_total = np.array([tm["beta"][f"beta[{h}]"]["mean"] for h in HAPLOTYPES])
    b = r["buffering"]
    rows.append(dict(
        gene_id=r["gene_id"],
        gene_type="no_ase" if not t["has_ase"] else ("ase_buff" if t["has_buffering"] else "ase_no_buff"),
        true_lambda=lam,
        lambda_hat=b["lambda"],
        lo=b["lo"],
        hi=b["hi"],
        bounded=b["bounded"],
        chi2_p=b["chi2_p"],
        p_gof=b["p_gof"],
        Q_ase=mahalanobis(np.asarray(r["ase_model"]["beta"]) - beta, r["ase_model"]["cov_beta"]),
        Q_total=mahalanobis(beta_total - lam * beta, tm["beta_cov"]),
    ))
genes = pl.DataFrame(rows)

summary = genes.group_by("gene_type").agg(
    n=pl.len(),
    frac_p_lt_0_05=(pl.col("chi2_p") < 0.05).mean(),
    frac_gof_p_lt_0_05=(pl.col("p_gof") < 0.05).mean(),
    median_lambda_error=(pl.col("lambda_hat") - pl.col("true_lambda")).median(),
    median_abs_lambda_error=(pl.col("lambda_hat") - pl.col("true_lambda")).abs().median(),
    ci_coverage=((pl.col("lo") <= pl.col("true_lambda")) & (pl.col("true_lambda") <= pl.col("hi"))).mean(),
    frac_ci_bounded=pl.col("bounded").mean(),
    corr_lambda=pl.corr("lambda_hat", "true_lambda", method="spearman"),
    Q_ase_mean_ratio=pl.col("Q_ase").mean() / 7,
    Q_ase_median_ratio=pl.col("Q_ase").median() / CHI2_7_MEDIAN,
    frac_Q_ase_gt_q95=(pl.col("Q_ase") > CHI2_7_Q95).mean(),
    Q_total_mean_ratio=pl.col("Q_total").mean() / 7,
    Q_total_median_ratio=pl.col("Q_total").median() / CHI2_7_MEDIAN,
    frac_Q_total_gt_q95=(pl.col("Q_total") > CHI2_7_Q95).mean(),
).with_columns(
    # lambda is undefined without ASE (beta = 0)
    pl.when(pl.col("gene_type") != "no_ase").then(pl.col(c)).alias(c)
    for c in ["median_lambda_error", "median_abs_lambda_error", "ci_coverage", "corr_lambda"]
).sort(pl.col("gene_type").replace_strict({"no_ase": 0, "ase_no_buff": 1, "ase_buff": 2}))

out = pathlib.Path(args.outfile)
genes.write_parquet(out.with_suffix(".genes.parquet"))
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3, fmt_str_lengths=60):
    text = (
        "Buffering pipeline on simulated data, by gene type\n"
        "(frac_p_lt_0_05: test of lambda = 1; should be 0.05 except for ase_buff (power).\n"
        " Q ratios 1 and frac_Q_gt_q95 0.05 when the beta covariances are calibrated;\n"
        " corr_lambda is only meaningful for ase_buff, where lambda varies.)\n"
        f"Genes with errors (excluded): {n_errors}\n\n"
        f"{summary.transpose(include_header=True, column_names='gene_type')}\n"
    )
print(text)
out.write_text(text)
