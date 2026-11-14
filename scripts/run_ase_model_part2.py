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
from ase_buffering.ase_model_2steps import (
    fit_stage2_eb,
    fit_stage2_sandwich,
    prepare_gene_data,
    stage1_draws,
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
CONFIG = dict(
    eta_prior_sd=3.0,
    min_eta_info=2.0,  # mice with less plug-in information are not fit for eta
    min_n_informative=50,
    max_abs_skew=0.5,  # flag mice whose stage-1 posterior is clearly non-Gaussian
)
# Also run the previous stage-2 method (full stage-1 covariance, inverted) and report
# it alongside, for comparison. Set to False once the sandwich method is settled.
RUN_EB_COMPARISON = True


def tolist(x):
    return np.asarray(x).tolist()


def eta_convergence(idata):
    """ESS / R-hat of the stage-1 etas (the draws stage 2 is built from)."""
    try:
        post = idata.posterior
        post = post.to_dataset() if hasattr(post, "to_dataset") else post
        ess_bulk = az.ess(post, var_names=["eta"], method="bulk")["eta"].values
        ess_tail = az.ess(post, var_names=["eta"], method="tail")["eta"].values
        rhat = az.rhat(post, var_names=["eta"])["eta"].values
        return dict(
            eta_ess_bulk=tolist(ess_bulk),
            eta_ess_tail=tolist(ess_tail),
            eta_rhat=tolist(rhat),
            min_ess_bulk_eta=float(ess_bulk.min()),
            median_ess_bulk_eta=float(np.median(ess_bulk)),
            min_ess_tail_eta=float(ess_tail.min()),
            max_rhat_eta=float(rhat.max()),
        )
    except Exception as e:
        return dict(convergence_error=f"{type(e).__name__}: {e}")

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

        ###### ASE MODEL - STAGE 1 -> 2 HANDOFF
        # Likelihood for eta (prior divided out exactly), per-mouse and full covariance
        handoff = stage1_draws(idata_ase1, gene_data, config=CONFIG)

        ###### ASE MODEL - STAGE 2 (sandwich variance)
        stage2 = fit_stage2_sandwich(gene_data, handoff)

        fit_idx = handoff["index"]
        result = dict(
            gene_id=gene_id,
            # Primary results: consumed by run_buffering_model.py / summarize_ase_models
            sigma_u=float(stage2["sigma_u"]),
            sigma_gamma=float(stage2["sigma_gamma"]),
            beta=tolist(stage2["beta"]),
            cov_beta=tolist(stage2["cov_beta"]),
            gamma=tolist(stage2["gamma"]),
            cov_gamma=tolist(stage2["cov_gamma"]),
            delta=tolist(stage2["delta"]),
            cov_delta=tolist(stage2["cov_delta"]),
            converged=bool(stage2["converged"]),
            stage2_method="sandwich",
            # Model-based covariance under the working (diagonal stage-1) covariance.
            # cov_beta / cov_beta_working >> 1 means cross-mouse correlation matters.
            cov_beta_working=tolist(stage2["cov_beta_working"]),
            diagnostics=dict(
                **stage2["diagnostics"],
                n_samples=len(gene_data.ids),
                n_het=int((gene_data.sign != 0).sum()),
                n_fit=len(fit_idx),
                n_draws=int(handoff["n_draws"]),
                min_eig_prior_division=handoff["min_eig_prior_division"],
                mean_abs_corr_eta=handoff["mean_abs_corr"],
                max_abs_corr_eta=handoff["max_abs_corr"],
                n_flagged_skew=len(handoff["flagged"]),
                max_abs_skew=float(np.abs(handoff["skew"]).max()),
                **eta_convergence(idata_ase1),
            ),
            # Per-mouse stage-1 handoff (in fit_mouse_ids order)
            per_mouse=dict(
                fit_mouse_ids=[gene_data.ids[i] for i in fit_idx],
                diplotype=[
                    gene_data.haplotypes[gene_data.hap1[i]]
                    + gene_data.haplotypes[gene_data.hap2[i]]
                    for i in fit_idx
                ],
                log_eff_ratio=tolist(gene_data.log_eff_ratio[fit_idx]),
                plugin_info=tolist(gene_data.info[fit_idx]),
                eta_post_mean=tolist(handoff["post_mean"]),
                eta_post_var=tolist(handoff["post_var"]),
                eta_lik_mean=tolist(handoff["mean"]),
                eta_lik_var=tolist(handoff["var_diag"]),
                skew=tolist(handoff["skew"]),
                flagged_skew=handoff["flagged"],
            ),
            error="none",
        )

        ###### COMPARISON: previous stage-2 method
        if RUN_EB_COMPARISON:
            try:
                lik_eb = stage1_likelihood(
                    idata_ase1,
                    gene_data,
                    config=dict(
                        stage2_cov="full",
                        cov_shrink=0.0,
                        max_abs_skew=CONFIG["max_abs_skew"],
                        eta_prior_sd=CONFIG["eta_prior_sd"],
                        min_eta_info=CONFIG["min_eta_info"],
                    ),
                )
                eb = fit_stage2_eb(gene_data, lik_eb)
                result["eb_comparison"] = dict(
                    sigma_u=float(eb["sigma_u"]),
                    sigma_gamma=float(eb["sigma_gamma"]),
                    beta=tolist(eb["beta"]),
                    cov_beta=tolist(eb["cov_beta"]),
                    converged=bool(eb["converged"]),
                )
            except Exception as e:
                result["eb_comparison"] = dict(error=f"{type(e).__name__}: {e}")

        results.append(result)

    except Exception as e:
        print(f"Encountered an error in {gene_id}:\n{e}")
        results.append({"gene_id": gene_id, "error": f"Exception: {e}"})

with open(OUTFILE, "wt") as f:
    json.dump(results, f)
