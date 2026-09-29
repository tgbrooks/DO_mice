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
    "ENSMUSG00000006576",
    "ENSMUSG00000000938",
    "ENSMUSG00000047045",
    "ENSMUSG00000024691",
    "ENSMUSG00000018585",
    "ENSMUSG00000005102",
    "ENSMUSG00000028690",
    "ENSMUSG00000037204",
    "ENSMUSG00000023027",
    "ENSMUSG00000070520",
    "ENSMUSG00000019969",
    "ENSMUSG00000045045",
    "ENSMUSG00000060679",
    "ENSMUSG00000067722",
    "ENSMUSG00000030036",
    "ENSMUSG00000034892",
    "ENSMUSG00000002102",
    "ENSMUSG00000032298",
    "ENSMUSG00000036309",
    "ENSMUSG00000024163",
    "ENSMUSG00000057551",
    "ENSMUSG00000025592",
    "ENSMUSG00000041354",
    "ENSMUSG00000040952",
    "ENSMUSG00000030602",
    "ENSMUSG00000036246",
    "ENSMUSG00000044795",
    "ENSMUSG00000032609",
    "ENSMUSG00000033184",
    "ENSMUSG00000031672",
    "ENSMUSG00000038766",
    "ENSMUSG00000042079",
    "ENSMUSG00000073176",
    "ENSMUSG00000059436",
    "ENSMUSG00000026096",
    "ENSMUSG00000075318",
    "ENSMUSG00000025940",
    "ENSMUSG00000013465",
    "ENSMUSG00000022897",
    "ENSMUSG00000039285",
    "ENSMUSG00000035944",
    "ENSMUSG00000033701",
    "ENSMUSG00000037119",
    "ENSMUSG00000026211",
    "ENSMUSG00000004661",
    "ENSMUSG00000001865",
    "ENSMUSG00000006906",
    "ENSMUSG00000069844",
    "ENSMUSG00000020608",
    "ENSMUSG00000034602",
]
# expr = allele_unique.group_by("gene_id").agg(pl.col("total_reads").median())
# expr.filter(pl.col("total_reads") > 50)

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
    pheno = all_pheno.with_columns(
        pl.col("DOwave").cast(str).cast(pl.Enum([str(x) for x in range(1, 6)]))
    ).join(size_factors, "mouse_id")
    total_model = make_total_model(gene_id, gene_class_counts, diplotypes, pheno)
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
    ase_summary = summarize_ase_model(idata_ase)
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
