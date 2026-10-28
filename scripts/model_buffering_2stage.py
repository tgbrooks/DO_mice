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
from ase_buffering.total_counts_model import make_total_model, summarize_total_model
from ase_buffering.covariance_deming_regression import covariance_deming_regression
from ase_buffering.ase_model_2steps import (
    make_stage1_model,
    fit_stage2_eb,
    prepare_gene_data,
    stage1_likelihood,
    summarize_ase_models,
)


parser = argparse.ArgumentParser("Run the models on the given genes")
parser.add_argument("--tissue", help="which tissue to use")
parser.add_argument("--outfile", help="which file to write json output to")
parser.add_argument("--gene_ids_file", help="file with one gene id per line")
parser.add_argument("--cores", type=int, help="how many cores to use", default=4)
parser.add_argument("--chains", type=int, help="how many chains to make", default=4)
parser.add_argument(
    "--ase_config",
    type=str,
    help="json dict as string of ASE model config",
    default=None,
)
args = parser.parse_args()

TISSUE = "Adipose"
OUTFILE = "temp.json"
TISSUE = args.tissue
OUTFILE = args.outfile
CORES = args.cores
CHAINS = args.chains
ASE_CONFIG = json.loads(args.ase_config) if args.ase_config else None
assert isinstance(ASE_CONFIG, dict)

gene_ids = [x.strip() for x in open(args.gene_ids_file).read().splitlines()]
print(f"Running on {len(gene_ids)} genes (gene_ids={gene_ids[:5]}...")

config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
HAPLOTYPES = config["haplotypes"].split(",")
HAP_TO_HAPNUM = {hap: i for i, hap in enumerate(HAPLOTYPES)}
SEX_TO_NUM = {"F": 0, "M": 1}
OUTLIER_MOUSE_IDS = config["outlier_ids"]

# gene_ids = [
#    "ENSMUSG00000053062",  # Jam2
#    "ENSMUSG00000040548",  # Tex2
# ]
# gene_ids = [  # randomly chosen genes with > 50 median total reads
#    "ENSMUSG00000038429",
#    "ENSMUSG00000060121",
#    "ENSMUSG00000060002",
#    "ENSMUSG00000030605",
#    "ENSMUSG00000032498",
#    "ENSMUSG00000047037",
#    "ENSMUSG00000003617",
#    "ENSMUSG00000071648",
#    "ENSMUSG00000001173",
#    "ENSMUSG00000079508",
#    "ENSMUSG00000042717",
#    "ENSMUSG00000025155",
#    "ENSMUSG00000026072",
#    "ENSMUSG00000020078",
#    "ENSMUSG00000032412",
#    "ENSMUSG00000055912",
#    "ENSMUSG00000022312",
#    "ENSMUSG00000033900",
#    "ENSMUSG00000001576",
#    "ENSMUSG00000063052",
#    "ENSMUSG00000053110",
#    "ENSMUSG00000039485",
#    "ENSMUSG00000021702",
#    "ENSMUSG00000051864",
#    "ENSMUSG00000034958",
#    "ENSMUSG00000053012",
#    "ENSMUSG00000036246",
#    "ENSMUSG00000001366",
#    "ENSMUSG00000025184",
#    "ENSMUSG00000005881",
#    "ENSMUSG00000023143",
#    "ENSMUSG00000042694",
#    "ENSMUSG00000074754",
#    "ENSMUSG00000035642",
#    "ENSMUSG00000059288",
#    "ENSMUSG00000027309",
#    "ENSMUSG00000020513",
#    "ENSMUSG00000085793",
#    "ENSMUSG00000026924",
#    "ENSMUSG00000032578",
#    "ENSMUSG00000035697",
#    "ENSMUSG00000037270",
#    "ENSMUSG00000009207",
#    "ENSMUSG00000030214",
#    "ENSMUSG00000004035",
#    "ENSMUSG00000025782",
#    "ENSMUSG00000000184",
#    "ENSMUSG00000051331",
#    "ENSMUSG00000044968",
# ]

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
        f"processed/Adipose/gbrs/{mouse_id}.compressed.h5",
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

        ###### ASE MODEL - STAGE 1

        stage1_config = {
            **dict(
                q_mode="adaptive_centered",  # "adaptive_centered", "constant_sigma_q" or "q_hat"
                sigma_q=0.3,
                sigma_q_tiny=2.0,
                sigma_q_sd=0.025,
                min_counts_to_center=50,
                eta_prior_sd=3.0,  # weak independent prior on each mouse's eta (divided out later)
                min_eta_info=2.0,  # mice with less plug-in information are not fit for eta
                min_n_informative=50,
                sigma_log_total_reads=1.0,
                target_accept=0.8,
            ),
            **ASE_CONFIG,
        }
        ase_model1 = make_stage1_model(
            gene_data,
            stage1_config,
        )

        _start = time.time()
        with ase_model1:
            idata_ase1 = pm.sample(
                500,
                cores=CORES,
                chains=CHAINS,
                random_seed=100,
                nuts_sampler="nutpie",
                quiet=quiet,
                target_accept=stage1_config["target_accept"],
            )
        _end = time.time()
        ase_model_time1 = _end - _start

        handoff = stage1_likelihood(
            idata_ase1,
            gene_data,
            config=dict(
                stage2_cov="full",
                cov_shrink=0.1,  # shrink stage-1 posterior correlations toward 0 (full mode)
                max_abs_skew=0.5,  # flag mice whose stage-1 posterior is clearly non-Gaussian
                eta_prior_sd=stage1_config["eta_prior_sd"],
                min_eta_info=stage1_config["min_eta_info"],
            ),
        )

        ###### ASE MODEL - STAGE 2
        stage2 = fit_stage2_eb(gene_data, handoff)

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
        )
        _start = time.time()
        with total_model:
            idata_total = pm.sample(
                500,
                cores=CORES,
                chains=CHAINS,
                random_seed=101,
                nuts_sampler="nutpie",
                quiet=quiet,
            )
        _end = time.time()
        total_model_time = _end - _start

        ##### BUFFERING MODEL
        beta_ase = stage2["beta"]
        cov_ase = stage2["cov_beta"]
        beta_total_draws = az.extract(idata_total, "posterior")["beta"].to_numpy()
        cov_total = np.cov(beta_total_draws)
        beta_total = np.mean(beta_total_draws, axis=1)
        buffering_res = covariance_deming_regression(
            beta_ase, beta_total, cov_ase, cov_total
        )

        # Report results
        ase_summary = summarize_ase_models(idata_ase1, stage2, gene_data)
        total_summary = summarize_total_model(idata_total)

        results.append(
            {
                "gene_id": gene_id,
                "ase_model": ase_summary,
                "total_model": total_summary,
                "buffering": buffering_res,
                "runtime": time.time() - _gene_start,
                "error": "none",
            }
        )
    except Exception as e:
        print(f"Encountered an error in {gene_id}:\n{e}")
        results.append({"gene_id": gene_id, "error": f"Exception: {e}"})

with open(OUTFILE, "wt") as outfile:
    json.dump(results, outfile)
