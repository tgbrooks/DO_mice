import polars as pl
import polars_bio as pb
import numpy as np
import pymc as pm
import arviz as az

import time
import yaml
import json

from util.compressed_emase import load_compressed_emase
from util.compat_classes import get_gene_class_counts
from ase_buffering.ase_model import make_ase_model, summarize_ase_model
from ase_buffering.total_counts_model import make_total_model, summarize_total_model
from ase_buffering.covariance_deming_regression import covariance_deming_regression

TISSUE = "Adipose"
OUTFILE = "temp.json"

config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
HAPLOTYPES = config["haplotypes"].split(",")
HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}
SEX_TO_NUM = {"F": 0, "M": 1}
OUTLIER_MOUSE_IDS = config["outlier_ids"]

gene_ids = [
    "ENSMUSG00000053062",  # Jam2
    "ENSMUSG00000040548",  # Tex2
]
gene_ids = [  # randomly chosen genes with > 50 median total reads
    "ENSMUSG00000038429",
    "ENSMUSG00000060121",
    "ENSMUSG00000060002",
    "ENSMUSG00000030605",
    "ENSMUSG00000032498",
    "ENSMUSG00000047037",
    "ENSMUSG00000003617",
    "ENSMUSG00000071648",
    "ENSMUSG00000001173",
    "ENSMUSG00000079508",
    "ENSMUSG00000042717",
    "ENSMUSG00000025155",
    "ENSMUSG00000026072",
    "ENSMUSG00000020078",
    "ENSMUSG00000032412",
    "ENSMUSG00000055912",
    "ENSMUSG00000022312",
    "ENSMUSG00000033900",
    "ENSMUSG00000001576",
    "ENSMUSG00000063052",
    "ENSMUSG00000053110",
    "ENSMUSG00000039485",
    "ENSMUSG00000021702",
    "ENSMUSG00000051864",
    "ENSMUSG00000034958",
    "ENSMUSG00000053012",
    "ENSMUSG00000036246",
    "ENSMUSG00000001366",
    "ENSMUSG00000025184",
    "ENSMUSG00000005881",
    "ENSMUSG00000023143",
    "ENSMUSG00000042694",
    "ENSMUSG00000074754",
    "ENSMUSG00000035642",
    "ENSMUSG00000059288",
    "ENSMUSG00000027309",
    "ENSMUSG00000020513",
    "ENSMUSG00000085793",
    "ENSMUSG00000026924",
    "ENSMUSG00000032578",
    "ENSMUSG00000035697",
    "ENSMUSG00000037270",
    "ENSMUSG00000009207",
    "ENSMUSG00000030214",
    "ENSMUSG00000004035",
    "ENSMUSG00000025782",
    "ENSMUSG00000000184",
    "ENSMUSG00000051331",
    "ENSMUSG00000044968",
]

size_factors = pl.read_csv(f"processed/{TISSUE}/size_factors.txt", separator="\t")

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
diplotypes = allele_unique.select("mouse_id", "gene_id", "diplotype")

all_pheno = pl.read_csv("phenotypes.csv.gz").rename({"mouse.id": "mouse_id"})

mouse_ids = sorted(allele_unique["mouse_id"].unique())

all_compatibility_classes = {}
for mouse_id in mouse_ids:
    if mouse_id in OUTLIER_MOUSE_IDS:
        continue
    all_compatibility_classes[mouse_id] = load_compressed_emase(
        f"processed/Adipose/gbrs/{mouse_id}.compressed.h5",
        [f"h{i}" for i in range(len(HAPLOTYPES))],
    )

quiet = len(gene_ids) > 1
results = []
for gene_id in gene_ids:
    _gene_start = time.time()
    print(f"Modelling {gene_id}")
    gene_class_counts = get_gene_class_counts(
        gene_id,
        tx_annot,
        all_compatibility_classes,
    )

    ###### ASE MODEL
    ase_model = make_ase_model(gene_id, gene_class_counts, diplotypes)
    _start = time.time()
    with ase_model:
        idata_ase = pm.sample(
            500,
            random_seed=100,
            nuts_sampler="nutpie",
            quiet=quiet,
        )
    _end = time.time()
    ase_model_time = _end - _start

    ###### TOTAL COUNTS MODEL
    sample_info = all_pheno.with_columns(
        pl.col("DOwave").cast(str).cast(pl.Enum([str(x) for x in range(1, 6)]))
    ).join(size_factors, "mouse_id")
    total_model = make_total_model(
        gene_id,
        gene_class_counts,
        diplotypes,
        sample_info,
    )
    _start = time.time()
    with total_model:
        idata_total = pm.sample(
            500,
            random_seed=101,
            nuts_sampler="nutpie",
            quiet=quiet,
        )
    _end = time.time()
    total_model_time = _end - _start

    ##### BUFFERING MODEL
    beta_ase_draws = az.extract(idata_ase, "posterior")["beta"].to_numpy()
    beta_total_draws = az.extract(idata_total, "posterior")["beta"].to_numpy()
    cov_ase = np.cov(beta_ase_draws)
    cov_total = np.cov(beta_total_draws)
    beta_ase = np.mean(beta_ase_draws, axis=1)
    beta_total = np.mean(beta_total_draws, axis=1)
    buffering_res = covariance_deming_regression(
        beta_ase, beta_total, cov_ase, cov_total
    )

    # Report results
    ase_summary = summarize_ase_model(idata_ase, gene_class_counts)
    ase_summary["runtime"] = ase_model_time
    total_summary = summarize_total_model(idata_total)
    ase_summary["runtime"] = total_model_time

    results.append(
        {
            "gene_id": gene_id,
            "ase_model": ase_summary,
            "total_model": total_summary,
            "buffering": buffering_res,
            "runtime": time.time() - _gene_start,
        }
    )

with open(OUTFILE, "wt") as outfile:
    json.dump(results, outfile)
