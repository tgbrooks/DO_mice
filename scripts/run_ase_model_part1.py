import polars as pl
import polars_bio as pb
import numpy as np
import pymc as pm

import argparse
import time
import yaml
import json

from util.compressed_emase import load_compressed_emase
from util.compat_classes import get_gene_class_counts
from util.netcdf import save_results, GeneError
from ase_buffering.ase_model_2steps import (
    make_stage1_model,
    prepare_gene_data,
)


parser = argparse.ArgumentParser("Run the models on the given genes")
parser.add_argument("--tissue", help="which tissue to use")
parser.add_argument(
    "--outfile", help="which file to write the netcdf format results to"
)
parser.add_argument("--gene_ids_file", help="file with one gene id per line")
parser.add_argument("--cores", type=int, help="how many cores to use", default=4)
parser.add_argument("--chains", type=int, help="how many chains to make", default=4)
parser.add_argument(
    "--config",
    type=str,
    help="json dict as string of ASE model config",
    default=None,
)
parser.add_argument(
    "--oracle",
    choices=["none", "alpha", "q", "both"],
    default="none",
    help="simulated data only: replace estimated alpha and/or q with the true values",
)
parser.add_argument(
    "--ref_tissue",
    default="Adipose",
    help="with --oracle: the real tissue the simulation was based on (source of true q)",
)
args = parser.parse_args()

TISSUE = "Adipose"
OUTFILE = "temp.json"
TISSUE = args.tissue
OUTFILE = args.outfile
CORES = args.cores
CHAINS = args.chains
CONFIG = json.loads(args.config) if args.config else {}
assert isinstance(CONFIG, dict)

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

ORACLE = args.oracle
if ORACLE != "none":
    # Truth as simulate_classes.py defines it:
    #   q     = q_hat of the real reference-tissue data (min_genotype_confidence=0.9)
    #   alpha = per sample x class values saved in sample_truths.parquet
    # Both are in the reference data's class order and are mapped to this data's
    # classes by class name.
    REF_TISSUE = args.ref_tissue
    sample_truths = pl.read_parquet(f"processed/{TISSUE}/sample_truths.parquet")
    ref_effective_lengths = pl.read_parquet(
        f"processed/{REF_TISSUE}/effective_lengths.by_gene.parquet"
    )
    # Needed for both oracles: true q, and the class order the true alpha is stored in
    ref_compatibility_classes = {}
    for mouse_id in sorted(
        pl.read_parquet(f"processed/{REF_TISSUE}/allele_unique_reads.parquet")[
            "mouse_id"
        ].unique()
    ):
        if mouse_id in config["outlier_ids"][REF_TISSUE]:
            continue
        ref_compatibility_classes[mouse_id] = load_compressed_emase(
            f"processed/{REF_TISSUE}/gbrs/{mouse_id}.compressed.h5",
            [f"h{i}" for i in range(len(HAPLOTYPES))],
        )

    def apply_oracle(gene_id, gene_data):
        """Overwrite gene_data.alpha and/or return the true q (else None).

        gene_data.info (and so which mice are fit for eta) is deliberately left as
        computed from the estimated alpha, so part 2 selects the same mice.
        """
        q_true, ref_names = load_ref(gene_id)
        # The simulated classes come from the reference compat matrix, so the class
        # sets must match; map by name since the class order can differ.
        mismatch = set(gene_data.class_names) ^ set(ref_names)
        if mismatch:
            raise ValueError(f"Class mismatch with reference data: {sorted(mismatch)}")
        perm = [ref_names.index(n) for n in gene_data.class_names]
        q_fixed = None
        if ORACLE in ("alpha", "both"):
            truth = sample_truths.filter(gene_id=gene_id)
            if truth.height == 0:
                raise ValueError(f"No sample truths for {gene_id}")
            true_alpha = {
                m: np.asarray(a)[perm]
                for m, a in zip(truth["mouse_id"], truth["alpha"].to_list())
            }
            gene_data.alpha = np.stack([true_alpha[m] for m in gene_data.ids])
        if ORACLE in ("q", "both"):
            q_fixed = q_true[:, perm]
        return q_fixed

    def load_ref(gene_id):
        """Reference-tissue gene data, built exactly as simulate_classes.py does."""
        gcc = get_gene_class_counts(gene_id, tx_annot, ref_compatibility_classes)
        ref = prepare_gene_data(
            gene_id,
            gcc,
            genotypes,
            ref_effective_lengths.filter(gene_id=gene_id).select(
                "mouse_id", "haplotype", "effective_length"
            ),
            config=dict(min_genotype_confidence=0.9, drop_all_zeros=True),
        )
        return ref.q_hat, ref.class_names

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
            **CONFIG,
        }
        if ORACLE != "none":
            q_fixed = apply_oracle(gene_id, gene_data)
            if q_fixed is not None:
                stage1_config["q_fixed"] = q_fixed
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
        # Record the stage-1 settings part 2 must use (prior division, fit-mouse mask)
        for key in ("eta_prior_sd", "min_eta_info", "sigma_q"):
            idata_ase1.posterior.attrs[f"stage1_{key}"] = float(stage1_config[key])

        results[gene_id] = idata_ase1
    except Exception as e:
        results[gene_id] = GeneError(e)
        print(f"Error in gene {gene_id}: {e}")
save_results(results, OUTFILE)
