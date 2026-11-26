"""
Summarize the paralog compensation fits of one chunk into a JSON list, one entry per
ordered pair (A = locus gene, B = response gene):
    theta, cov_theta       effect of A's locus on B (from the samples)
    beta_a, cov_beta_a     A's haplotype effects on A's own expression (total-counts
                           model, from the buffering results)
    compensation           theta = kappa * beta_a by Deming-type regression: kappa (the
                           elasticity of B w.r.t. A; < 0 = compensation) with a profile
                           interval, the likelihood-ratio test of kappa = 0, and the
                           goodness of fit of a single kappa
    theta_test             chi2 test of theta = 0 (any effect of A's locus on B)
    beta_a_test            chi2 test of beta_a = 0 (strength of A's own genetic effects)
"""

import argparse
import json

import numpy as np
import polars as pl
import scipy.stats
import yaml

from util.netcdf import load_result
from ase_buffering.paralog_model import summarize_paralog_model
from ase_buffering.covariance_deming_regression import covariance_deming_regression

KAPPA_GRID = np.linspace(-3, 3, 1201)

parser = argparse.ArgumentParser()
parser.add_argument("--samples", help=".nc file from run_paralog_model.py")
parser.add_argument("--chunk", help="chunk TSV: pair_id, pair_type, gene_a, gene_b")
parser.add_argument("--buffering", help="results/{tissue}/buffering_full_results.parquet")
parser.add_argument("--outfile")
args = parser.parse_args()

HAPLOTYPES = yaml.safe_load(open("config.yaml"))["haplotypes"].split(",")


def chi2_test(b, V):
    """b' V^+ b over the directions V supports (sum-to-zero effects: rank n - 1)."""
    w, U = np.linalg.eigh(np.asarray(V))
    k = w > w.max() * 1e-8
    stat = float(np.sum((U[:, k].T @ b) ** 2 / w[k]))
    df = int(k.sum())
    return dict(chi2=stat, df=df, p=float(scipy.stats.chi2.sf(stat, df)))


chunk = pl.read_csv(args.chunk, separator="\t")
buffering = (
    pl.scan_parquet(args.buffering)
    .filter(pl.col("gene_id").is_in(list(set(chunk["gene_a"]))), pl.col("error") == "none")
    .select("gene_id", "total_model")
    .collect()
)
total_beta = {}
for row in buffering.to_dicts():
    tm = row["total_model"]
    total_beta[row["gene_id"]] = (
        np.array([tm["beta"][f"beta[{h}]"]["mean"] for h in HAPLOTYPES]),
        np.asarray(tm["beta_cov"]),
    )

results = []
for pair in chunk.to_dicts():
    out = dict(pair)
    idata, err = load_result(args.samples, pair["pair_id"])
    if err is not None:
        results.append({**out, "error": f"{err['error_type']}: {err['error']}"})
        continue
    try:
        fit = summarize_paralog_model(idata)
        out.update(fit, runtime=float(idata.posterior.attrs["runtime_s"]))
        theta, V_theta = np.array(fit["theta"]), np.array(fit["cov_theta"])
        out["theta_test"] = chi2_test(theta, V_theta)
        if pair["gene_a"] not in total_beta:
            out.update(error="none", compensation_error="no total-model fit for gene A")
            results.append(out)
            continue
        beta_a, V_a = total_beta[pair["gene_a"]]
        out.update(beta_a=beta_a.tolist(), cov_beta_a=V_a.tolist())
        out["beta_a_test"] = chi2_test(beta_a, V_a)
        dem = covariance_deming_regression(beta_a, theta, V_a, V_theta, grid=KAPPA_GRID)
        # Q at kappa = 0 is theta' V_theta^+ theta: profile likelihood ratio for kappa = 0
        lr = max(out["theta_test"]["chi2"] - dem["Q"], 0.0)
        out["compensation"] = dict(
            kappa=dem["lambda"],
            kappa_lo=dem["lo"],
            kappa_hi=dem["hi"],
            kappa_bounded=dem["bounded"],
            kappa0_lr=lr,
            kappa0_p=float(scipy.stats.chi2.sf(lr, 1)),
            Q=dem["Q"],
            df_gof=dem["df_gof"],
            p_gof=dem["p_gof"],
        )
        out.update(error="none", compensation_error="none")
    except Exception as e:
        print(f"Encountered an error summarizing {pair['pair_id']}:\n{e}")
        out = {**dict(pair), "error": f"{type(e).__name__}: {e}"}
    results.append(out)

with open(args.outfile, "wt") as f:
    json.dump(results, f)
