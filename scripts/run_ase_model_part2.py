import polars as pl
import polars_bio as pb
import numpy as np

import argparse
import time
import yaml
import json

from util.compressed_emase import load_compressed_emase
from util.compat_classes import get_gene_class_counts
from util.netcdf import load_result
from ase_buffering.ase_model_2steps import (
    fit_stage2_eb,
    prepare_gene_data,
    stage1_likelihood,
)


parser = argparse.ArgumentParser("Run the models on the given genes")
parser.add_argument("--tissue", help="which tissue to use")
parser.add_argument("--part1", help="file containing part1 results (netcdf format)")
parser.add_argument("--outfile", help="which file to write json output to")
parser.add_argument("--gene_ids_file", help="file with one gene id per line")
parser.add_argument("--cores", type=int, help="how many cores to use", default=4)
parser.add_argument("--chains", type=int, help="how many chains to make", default=4)
args = parser.parse_args()

TISSUE = args.tissue
PART1_FILE = args.part1
OUTFILE = args.outfile
CORES = args.cores
CHAINS = args.chains
stage1_config = dict(
    eta_prior_sd=3.0,
    min_eta_info=2.0,  # mice with less plug-in information are not fit for eta
    min_n_informative=50,
)

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
        idata_ase1, err = load_result(PART1_FILE, gene_id)
        if err:
            print(f"Skipping {gene_id} due to error in part1")
            results.append(
                {"gene_id": gene_id, "error": f"Exception during part1: {err}"}
            )
            continue

        ###### ASE MODEL - STAGE 1
        handoff = stage1_likelihood(
            idata_ase1,
            gene_data,
            config=dict(
                stage2_cov="full",
                cov_shrink=0.1,  # shrink stage-1 posterior correlations toward 0 (full mode)
                max_abs_skew=0.5,  # flag mice whose stage-1 posterior is clearly non-Gaussian
                eta_prior_sd=CONFIG["eta_prior_sd"],
                min_eta_info=CONFIG["min_eta_info"],
            ),
        )

        ###### ASE MODEL - STAGE 2
        stage2 = fit_stage2_eb(gene_data, handoff)
        results.append(
            {
                "gene_id": gene_id,
                **stage2,
                "error": "none",
            }
        )

    except Exception as e:
        print(f"Encountered an error in {gene_id}:\n{e}")
        results.append({"gene_id": gene_id, "error": f"Exception: {e}"})

with open(OUTFILE, "wt") as f:
    json.dump(results, f)
