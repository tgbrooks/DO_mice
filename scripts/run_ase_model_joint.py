"""
Fit the joint ASE model (ase_buffering/ase_model_joint.py) to each gene and save the
posterior samples of all genes to one .nc file (util/netcdf.py), one group per gene.
Genes that fail or are skipped get an error group instead; skipped genes' messages
start with SKIPPED_PREFIX. The config, draws and runtime are stored in each gene's
posterior attrs. scripts/summarize_ase_model_joint.py turns the file into JSON.
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
from util.netcdf import GeneError, save_results
from util.sim_oracle import ORACLE_MODES, SimOracle
from ase_buffering.ase_model_2steps import prepare_gene_data
from ase_buffering.ase_model_joint import DEFAULTS, make_joint_model

parser = argparse.ArgumentParser("Fit the joint ASE model to the given genes")
parser.add_argument("--tissue", help="which tissue to use")
parser.add_argument("--outfile", help=".nc output file")
parser.add_argument("--gene_ids_file", help="file with one gene id per line")
parser.add_argument("--cores", type=int, default=4)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--draws", type=int, default=500)
parser.add_argument("--config", default=None, help="JSON dict overriding DEFAULTS")
parser.add_argument(
    "--oracle",
    choices=ORACLE_MODES,
    default="none",
    help="simulated data only: replace estimated alpha and/or q with the true values",
)
parser.add_argument(
    "--ref_tissue",
    default="Adipose",
    help="with --oracle: the real tissue the simulation was based on (source of true q)",
)
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

oracle = (
    SimOracle(
        args.oracle,
        sim_tissue=TISSUE,
        ref_tissue=args.ref_tissue,
        tx_annot=tx_annot,
        genotypes=genotypes,
        n_haplotypes=len(HAPLOTYPES),
        ref_outlier_ids=config["outlier_ids"][args.ref_tissue],
    )
    if args.oracle != "none"
    else None
)

SKIPPED_PREFIX = "SKIPPED: "


class SkippedGene(Exception):
    pass


quiet = len(gene_ids) > 1
results = {}
for gene_id in gene_ids:
    try:
        start = time.time()
        print(f"Modelling {gene_id}", flush=True)
        gene_class_counts = get_gene_class_counts(
            gene_id, tx_annot, all_compatibility_classes
        )
        median_expr = np.median(gene_class_counts.counts.sum(axis=1))
        if median_expr < 50:
            raise SkippedGene(f"{SKIPPED_PREFIX}median expression {median_expr} < 50")
        gene_data = prepare_gene_data(
            gene_id,
            gene_class_counts,
            genotypes,
            effective_lengths.filter(gene_id=gene_id).select(
                "mouse_id", "haplotype", "effective_length"
            ),
            config=dict(min_genotype_confidence=0.9),
        )
        model_config = dict(CONFIG)
        if oracle is not None:
            q_fixed = oracle.apply(gene_id, gene_data)  # also sets true alpha if asked
            if q_fixed is not None:
                model_config["q_fixed"] = q_fixed
        with make_joint_model(gene_data, model_config):
            idata = pm.sample(
                args.draws,
                cores=args.cores,
                chains=args.chains,
                random_seed=100,
                nuts_sampler="nutpie",
                quiet=quiet,
                target_accept=CONFIG["target_accept"],
            )
        idata.posterior.attrs["joint_config"] = json.dumps({**CONFIG, "oracle": args.oracle})
        idata.posterior.attrs["draws"] = args.draws
        idata.posterior.attrs["runtime_s"] = time.time() - start
        results[gene_id] = idata
    except Exception as e:
        print(f"{gene_id}: {e}")
        results[gene_id] = GeneError(e)

save_results(results, args.outfile)
