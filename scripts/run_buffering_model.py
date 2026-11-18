import polars as pl
import polars_bio as pb
import numpy as np
import arviz as az

import argparse
import time
import yaml
import json

from util.compressed_emase import load_compressed_emase
from util.compat_classes import get_gene_class_counts
from util.netcdf import load_result
from ase_buffering.total_counts_model import summarize_total_model
from ase_buffering.covariance_deming_regression import covariance_deming_regression
from ase_buffering.ase_model_2steps import prepare_gene_data


parser = argparse.ArgumentParser("Run the models on the given genes")
parser.add_argument("--tissue", help="which tissue to use")
parser.add_argument(
    "--ase", help="joint ASE model summary JSON (scripts/summarize_ase_model_joint.py)"
)
parser.add_argument("--total", help="file with results from total model")
parser.add_argument("--outfile", help="which file to write json output to")
parser.add_argument("--gene_ids_file", help="file with one gene id per line")
parser.add_argument("--cores", type=int, help="how many cores to use", default=4)
parser.add_argument("--chains", type=int, help="how many chains to make", default=4)
args = parser.parse_args()

TISSUE = args.tissue
OUTFILE = args.outfile
CORES = args.cores
CHAINS = args.chains
ASE_FILE = args.ase
TOTAL_FILE = args.total

gene_ids = [x.strip() for x in open(args.gene_ids_file).read().splitlines()]
print(f"Running on {len(gene_ids)} genes (gene_ids={gene_ids[:5]}...")

config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
HAPLOTYPES = config["haplotypes"].split(",")
HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}
SEX_TO_NUM = {"F": 0, "M": 1}
OUTLIER_MOUSE_IDS = config["outlier_ids"][TISSUE]

size_factors = pl.read_csv(f"processed/{TISSUE}/size_factors.txt", separator="\t")
effective_lengths = pl.read_parquet(
    f"processed/{TISSUE}/effective_lengths.by_gene.parquet"
)

gene_annot = (
    pb.scan_gtf(
        "gbrs_ref/v116/reference.gtf.gz",
        attr_fields=["gene_id", "gene_name", "gene_biotype"],
    )
    .filter(pl.col("type") == "gene")
    .collect()
)

tx_annot = (
    pb.scan_gtf(
        "gbrs_ref/v116/reference.gtf.gz",
        attr_fields=["gene_id", "transcript_id"],
    )
    .filter(pl.col("type") == "transcript")
    .collect()
)

allele_unique = pl.read_parquet(f"processed/{TISSUE}/allele_unique_reads.parquet")
genotypes = pl.read_parquet("processed/genotypes.parquet").select(
    "gene_id",
    "mouse_id",
    "genotype_confidence",
    diplotype="genotype",
)


all_pheno = pl.read_csv("phenotypes.csv.gz").rename({"mouse.id": "mouse_id"})

mouse_ids = sorted(allele_unique["mouse_id"].unique())

all_compatibility_classes = {}
for mouse_id in mouse_ids:
    if mouse_id in OUTLIER_MOUSE_IDS:
        continue
    all_compatibility_classes[mouse_id] = load_compressed_emase(
        f"processed/{TISSUE}/gbrs/{mouse_id}.compressed.h5",
        [f"h{i}" for i in range(len(HAPLOTYPES))],
    )

with open(ASE_FILE) as f:
    ase_results = {g["gene_id"]: g for g in json.load(f)}

quiet = len(gene_ids) > 1
results = []
for gene_id in gene_ids:
    try:
        _gene_start = time.time()
        print(f"Modelling {gene_id}", flush=True)
        gene_class_counts = get_gene_class_counts(
            gene_id,
            tx_annot,
            all_compatibility_classes,
        )
        gene_effective_lengths = effective_lengths.filter(gene_id=gene_id).select(
            "mouse_id", "haplotype", "effective_length"
        )
        median_expr = np.median(gene_class_counts.counts.sum(axis=1))
        if median_expr < 50:
            print(f"Skipping {gene_id} - median expression too low")
            continue

        ###### GENE DATA
        gene_data = prepare_gene_data(
            gene_id,
            gene_class_counts,
            genotypes,
            gene_effective_lengths,
            config=dict(
                min_genotype_confidence=0.9,
            ),
        )

        ##### BUFFERING MODEL
        ase = ase_results.get(gene_id)
        if ase is None or ase["error"] != "none":
            err = "missing from ASE results" if ase is None else ase["error"]
            print(f"Skipping {gene_id} due to ASE model: {err}")
            results.append({"gene_id": gene_id, "error": f"ASE model: {err}"})
            continue
        beta_ase = np.asarray(ase["beta"])
        cov_ase = np.asarray(ase["cov_beta"])

        idata_total, err_total = load_result(TOTAL_FILE, gene_id)
        if err_total is not None:
            print(f"Skipping {gene_id} due to error in total model: {err_total}")
            results.append(
                {"gene_id": gene_id, "error": f"Total model: {err_total['error']}"}
            )
            continue

        beta_total_draws = az.extract(idata_total, "posterior")["beta"].to_numpy()
        cov_total = np.cov(beta_total_draws)
        beta_total = np.mean(beta_total_draws, axis=1)
        buffering_res = covariance_deming_regression(
            beta_ase, beta_total, cov_ase, cov_total
        )

        # Report results
        ase_summary = {k: v for k, v in ase.items() if k not in ("gene_id", "error")}
        total_summary = summarize_total_model(idata_total)

        results.append(
            {
                "gene_id": gene_id,
                "ase_model": ase_summary,
                "total_model": total_summary,
                "mouse_ids_used": gene_data.ids,  # same order as alpha.shape[0] and info; only hets get a u in ase_model
                "gene_data": {
                    "alpha": gene_data.alpha,
                    "info": gene_data.info,
                    "q_hat": gene_data.q_hat,
                },
                "buffering": buffering_res,
                "runtime": time.time() - _gene_start,
                "error": "none",
            }
        )
    except Exception as e:
        print(f"Encountered an error in {gene_id}:\n{e}")
        results.append({"gene_id": gene_id, "error": f"Exception: {e}"})

out = pl.DataFrame(results, infer_schema_length=None)
if "mouse_ids_used" in out.columns:
    out = out.with_columns(pl.col("mouse_ids_used").cast(pl.List(pl.Enum(mouse_ids))))
out.write_parquet(OUTFILE)
