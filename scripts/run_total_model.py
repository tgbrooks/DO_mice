import polars as pl
import polars_bio as pb
import numpy as np
import pymc as pm
import arviz as az

import argparse
import time
import yaml
import json

from util.compressed_emase import load_compressed_emase
from util.compat_classes import get_gene_class_counts
from util.netcdf import save_results, GeneError
from ase_buffering.total_counts_model import make_total_model


parser = argparse.ArgumentParser("Run the models on the given genes")
parser.add_argument("--tissue", help="which tissue to use")
parser.add_argument("--outfile", help="which file to write json output to")
parser.add_argument("--gene_ids_file", help="file with one gene id per line")
parser.add_argument("--cores", type=int, help="how many cores to use", default=4)
parser.add_argument("--chains", type=int, help="how many chains to make", default=4)
parser.add_argument(
    "--config",
    type=str,
    help="json dict as string of ASE model config",
    default=None,
)
args = parser.parse_args()

TISSUE = args.tissue
OUTFILE = args.outfile
CORES = args.cores
CHAINS = args.chains
ASE_CONFIG = json.loads(args.config) if args.config else {}
assert isinstance(ASE_CONFIG, dict)

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
results = {}
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

        ###### TOTAL COUNTS MODEL
        sample_info = all_pheno.with_columns(
            pl.col("DOwave").cast(str).cast(pl.Enum([str(x) for x in range(1, 6)]))
        ).join(size_factors, "mouse_id")
        total_model = make_total_model(
            gene_id,
            gene_class_counts,
            genotypes,
            sample_info,
            gene_effective_lengths,
            # Same genotype filter as the ASE model (prepare_gene_data)
            min_genotype_confidence=0.9,
        )
        with total_model:
            idata_total = pm.sample(
                500,
                cores=CORES,
                chains=CHAINS,
                random_seed=101,
                nuts_sampler="nutpie",
                quiet=quiet,
            )
        results[gene_id] = idata_total

    except Exception as e:
        print(f"Encountered an error in {gene_id}:\n{e}")
        results[gene_id] = GeneError(e)

save_results(results, OUTFILE)
