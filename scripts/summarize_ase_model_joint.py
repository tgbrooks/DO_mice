"""
Summarize the joint ASE model's saved samples (from scripts/run_ase_model_joint.py)
into a JSON list of per-gene results: beta, cov_beta, QC quantities and convergence
diagnostics, or {"gene_id", "error"} for genes that failed. Skipped genes are omitted.
"""

import argparse
import json

from util.netcdf import load_result
from ase_buffering.ase_model_joint import summarize_joint_model

SKIPPED_PREFIX = "SKIPPED: "  # as in scripts/run_ase_model_joint.py

parser = argparse.ArgumentParser("Summarize joint ASE model samples")
parser.add_argument("--samples", help=".nc file written by run_ase_model_joint.py")
parser.add_argument("--gene_ids_file", help="file with one gene id per line")
parser.add_argument("--outfile", help="JSON output file")
args = parser.parse_args()

gene_ids = [x.strip() for x in open(args.gene_ids_file).read().splitlines()]
results = []
for gene_id in gene_ids:
    idata, err = load_result(args.samples, gene_id)
    if err is not None:
        if err["error"].startswith(SKIPPED_PREFIX):
            continue
        results.append({"gene_id": gene_id, "error": f"{err['error_type']}: {err['error']}"})
        continue
    try:
        attrs = idata.posterior.attrs
        results.append(
            dict(
                gene_id=gene_id,
                **summarize_joint_model(idata),
                config=json.loads(attrs["joint_config"]),
                runtime=float(attrs["runtime_s"]),
                error="none",
            )
        )
    except Exception as e:
        print(f"Encountered an error summarizing {gene_id}:\n{e}")
        results.append({"gene_id": gene_id, "error": f"{type(e).__name__}: {e}"})

with open(args.outfile, "wt") as f:
    json.dump(results, f)
