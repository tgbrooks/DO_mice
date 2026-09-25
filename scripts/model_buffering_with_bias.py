import polars as pl
import polars_bio as pb
import numpy as np
import pymc as pm
import arviz as az
import time

import yaml

from util.compressed_emase import load_compressed_emase
from util.compat_classes import get_gene_class_counts
from ase_buffering.ase_model import make_ase_model
from ase_buffering.total_counts_model import make_total_model
from ase_buffering.covariance_deming_regression import covariance_deming_regression

TISSUE = "Adipose"

config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
HAPLOTYPES = config["haplotypes"].split(",")
HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}
SEX_TO_NUM = {"F": 0, "M": 1}
OUTLIER_MOUSE_IDS = config["outlier_ids"]

gene_ids = [
    # "ENSMUSG00000053062", # Jam2
    "ENSMUSG00000040548",  # Tex2
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

for gene_id in gene_ids:
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
            # quiet=True,
        )
    _end = time.time()
    ase_model_time = _end - _start
    ase_summary = pl.DataFrame(az.summary(idata_ase).reset_index())

    ###### TOTAL COUNTS MODEL
    pheno = all_pheno.with_columns(
        pl.col("DOwave").cast(str).cast(pl.Enum([str(x) for x in range(1, 6)]))
    ).join(size_factors, "mouse_id")
    total_model = make_total_model(gene_id, gene_class_counts, diplotypes, pheno)
    _start = time.time()
    with total_model:
        idata_totals = pm.sample(
            500,
            random_seed=101,
            nuts_sampler="nutpie",
            # quiet=True,
        )
    _end = time.time()
    total_model_time = _end - _start
    total_summary = pl.DataFrame(
        az.summary(idata_totals, ["beta"], filter_vars="like").reset_index()
    )

    ##### BUFFERING MODEL
    beta_ase_draws = az.extract(idata_ase, "posterior")["beta"].to_numpy()
    beta_total_draws = az.extract(idata_totals, "posterior")["beta"].to_numpy()
    cov_ase = np.cov(beta_ase_draws)
    cov_total = np.cov(beta_total_draws)
    beta_ase = np.mean(beta_ase_draws, axis=1)
    beta_total = np.mean(beta_total_draws, axis=1)
    buffering_res = covariance_deming_regression(
        beta_ase, beta_total, cov_ase, cov_total
    )
