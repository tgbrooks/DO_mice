"""
Fit the joint ASE model (ase_buffering/ase_model_joint.py) to each gene and write a
JSON list of per-gene summaries: beta, cov_beta, QC quantities and convergence
diagnostics, or {"gene_id", "error"} for genes that failed.
"""

import argparse
import json
import time

import numpy as np
import polars as pl
import polars_bio as pb
import pymc as pm
import yaml

from util.compressed_emase import load_compressed_emase
from util.compat_classes import get_gene_class_counts
from ase_buffering.ase_model_2steps import prepare_gene_data
from ase_buffering.ase_model_joint import (
    DEFAULTS,
    make_joint_model,
    summarize_joint_model,
)

parser = argparse.ArgumentParser("Fit the joint ASE model to the given genes")
parser.add_argument("--tissue", help="which tissue to use")
parser.add_argument("--outfile", help="JSON output file")
parser.add_argument("--gene_ids_file", help="file with one gene id per line")
parser.add_argument("--cores", type=int, default=4)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--draws", type=int, default=500)
parser.add_argument("--config", default=None, help="JSON dict overriding DEFAULTS")
args = parser.parse_args()

TISSUE = args.tissue
CONFIG = {**DEFAULTS, **(json.loads(args.config) if args.config else {})}

gene_ids = [x.strip() for x in open(args.gene_ids_file).read().splitlines()]
print(f"Running on {len(gene_ids)} genes (gene_ids={gene_ids[:5]}...")

config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
HAPLOTYPES = config["haplotypes"].split(",")
OUTLIER_MOUSE_IDS = config["outlier_ids"][TISSUE]

effective_lengths = pl.read_parquet(
    f"processed/{TISSUE}/effective_lengths.by_gene.parquet"
)
tx_annot = (
    pb.scan_gtf(
        "gbrs_ref/v116/reference.gtf.gz",
        attr_fields=["gene_id", "transcript_id"],
    )
    .filter(pl.col("type") == "transcript")
    .collect()
)
genotypes = pl.read_parquet("processed/genotypes.parquet").select(
    "gene_id",
    "mouse_id",
    "genotype_confidence",
    diplotype="genotype",
)
mouse_ids = sorted(
    pl.read_parquet(f"processed/{TISSUE}/allele_unique_reads.parquet")[
        "mouse_id"
    ].unique()
)
all_compatibility_classes = {
    mouse_id: load_compressed_emase(
        f"processed/{TISSUE}/gbrs/{mouse_id}.compressed.h5",
        [f"h{i}" for i in range(len(HAPLOTYPES))],
    )
    for mouse_id in mouse_ids
    if mouse_id not in OUTLIER_MOUSE_IDS
}

quiet = len(gene_ids) > 1
results = []
for gene_id in gene_ids:
    try:
        start = time.time()
        print(f"Modelling {gene_id}", flush=True)
        gene_class_counts = get_gene_class_counts(
            gene_id, tx_annot, all_compatibility_classes
        )
        median_expr = np.median(gene_class_counts.counts.sum(axis=1))
        if median_expr < 50:
            print(f"Skipping {gene_id} - median expression too low")
            continue
        gene_data = prepare_gene_data(
            gene_id,
            gene_class_counts,
            genotypes,
            effective_lengths.filter(gene_id=gene_id).select(
                "mouse_id", "haplotype", "effective_length"
            ),
            config=dict(min_genotype_confidence=0.9),
        )
        with make_joint_model(gene_data, CONFIG):
            idata = pm.sample(
                args.draws,
                cores=args.cores,
                chains=args.chains,
                random_seed=100,
                nuts_sampler="nutpie",
                quiet=quiet,
                target_accept=CONFIG["target_accept"],
            )
        results.append(
            dict(
                gene_id=gene_id,
                **summarize_joint_model(idata, gene_data),
                config=CONFIG,
                runtime=time.time() - start,
                error="none",
            )
        )
    except Exception as e:
        print(f"Encountered an error in {gene_id}:\n{e}")
        results.append({"gene_id": gene_id, "error": f"{type(e).__name__}: {e}"})

with open(args.outfile, "wt") as f:
    json.dump(results, f)
