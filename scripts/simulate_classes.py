"""
Simulate data at the 'read class' counts level.
"""

import argparse
from typing import Any
import pathlib

import json
import yaml
import numpy as np
import polars as pl
import polars_bio as pb
import scipy.linalg
import scipy.sparse
import formulaic
from util.compat_classes import CompatClassesDf, get_gene_class_counts
from util.compressed_emase import load_compressed_emase, Emase, write_compressed_emase
from ase_buffering.ase_model_2steps import prepare_gene_data


parser = argparse.ArgumentParser("simulate read class counts data")
parser.add_argument("--outdir", help="directory to write all output to")
parser.add_argument("--tissue", help="Tissue to use as input")
parser.add_argument(
    "--gene_id_file", help="file containing one gene id per line of genes to simulate"
)

args = parser.parse_args()

TISSUE = args.tissue
outdir = pathlib.Path(args.outdir)
outdir.mkdir(exist_ok=True)

if args.gene_id_file:
    gene_ids = [x.strip() for x in open(args.gene_id_file).read().strip().splitlines()]
else:
    # A few test genes
    gene_ids = [
        x.strip()
        for x in """
ENSMUSG00000010025
ENSMUSG00000057649
""".strip().splitlines()
    ]

config: dict[str, Any] = dict(
    seed=1,
    gene_ids=gene_ids,
    # These values estimated from "Adipose" tissue 250 mouse run
    sigma_u_zero_frac=0.4,
    log_sigma_u_mean=-2.1,
    log_sigma_u_std=0.7,
    sigma_gamma_zero_frac=0.6,
    log_sigma_gamma_mean=-2.9,
    log_sigma_gamma_std=0.9,
    # Parameters not estimated from real data
    beta_frac_zero=0.33,
    beta_std=0.3,
    buffering_frac=0.5,
    buffering_factor_hi=0.5,
    buffering_factor_lo=0.5,
    sex_effect_std=0.1,
    DO_wave_effect_std=0.05,
)

rng = np.random.default_rng(config["seed"])

pipeline_config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
HAPLOTYPES = pipeline_config["haplotypes"].split(",")
n_haps = len(HAPLOTYPES)
OUTLIER_MOUSE_IDS = pipeline_config["outlier_ids"]

# --------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------


def sigma_total_for_target(mu0, alpha, phi):
    """
    Per-gene sd of log total-expression noise so that simulated totals match the
    total model's variance M + phi * M^2.

    mu0:   samples x classes expected counts without total-level noise
    alpha: samples x classes NB dispersion (var = mu + mu^2 / alpha)
    phi:   E[exp(log_disp)] from the total model

    Noise: mu = mu0 * Z, Z = exp(eps), eps ~ N(-s^2/2, s^2)  (so E[Z] = 1)
      Var(T) = M0 + e^{s^2} * sum_k mu0_k^2 / alpha_k + (e^{s^2} - 1) * M0^2
    Matching gives e^{s^2} = (1 + phi) / (1 + c),  c = sum_k mu0_k^2/alpha_k / M0^2.
    Returns (s, c_mean); s = 0 if class-level noise alone already exceeds the target.
    """
    M0 = mu0.sum(axis=1)
    keep = M0 > 0
    c = (mu0[keep] ** 2 / alpha[keep]).sum(axis=1) / M0[keep] ** 2
    c_mean = float(c.mean())
    ratio = (1 + phi) / (1 + c_mean)
    s = float(np.sqrt(np.log(ratio))) if ratio > 1 else 0.0
    return s, c_mean


# --------------------------------------------------------------------------------
# Load real data for reference
# --------------------------------------------------------------------------------

size_factors = pl.read_csv(f"processed/{TISSUE}/size_factors.txt", separator="\t")
effective_lengths = pl.read_parquet(
    f"processed/{TISSUE}/effective_lengths.by_gene.parquet"
)

tx_annot = (
    pb.scan_gtf(
        "gbrs_ref/v116/reference.gtf.gz",
        attr_fields=["gene_id", "transcript_id", "tag"],
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

all_pheno = (
    pl.read_csv("phenotypes.csv.gz")
    .rename({"mouse.id": "mouse_id"})
    .with_columns(
        DOwave=pl.col("DOwave").cast(str).cast(pl.Enum(list("12345"))),
    )
)
n_DO_waves = all_pheno["DOwave"].n_unique()

mouse_ids = sorted(allele_unique["mouse_id"].unique())

# --------------------------------------------------------------------------------
# Load model results
# --------------------------------------------------------------------------------


def load_model_results():
    error_genes = []
    out = []
    with open(f"results/{TISSUE}/buffering_with_bias.ndjson") as f:

        def get(field_str):
            entries = field_str.split(".")
            temp = record
            for entry in entries:
                temp = temp[entry]
            return temp

        for line in f:
            record = json.loads(line.strip())
            if record.get("error", "error") != "none":
                error_genes.append(record)
                continue

            gene_id = get("gene_id")
            if gene_id not in config["gene_ids"]:
                continue
            out.append(
                {
                    "gene_id": gene_id,
                    "total_model": {
                        "log_disp": get("total_model.core.log_disp.mean"),
                        "log_disp_se": get("total_model.core.log_disp.sd"),
                    },
                }
            )

    # print(f"Skipped {len(error_genes)} genes with errors")
    return out


buffering_with_bias_objs = load_model_results()
buffering_res_all = pl.DataFrame(buffering_with_bias_objs)
total_fits = buffering_res_all.select(
    "gene_id",
    log_disp=pl.col("total_model").struct.field("log_disp"),
    log_disp_se=pl.col("total_model").struct.field("log_disp_se"),
)

# --------------------------------------------------------------------------------
# Load class count data for all mice + genes
# --------------------------------------------------------------------------------

all_compatibility_classes = {}
for mouse_id in mouse_ids:
    if mouse_id in OUTLIER_MOUSE_IDS:
        continue
    all_compatibility_classes[mouse_id] = load_compressed_emase(
        f"processed/{TISSUE}/gbrs/{mouse_id}.compressed.h5",
        [f"h{i}" for i in range(len(HAPLOTYPES))],
    )


# --------------------------------------------------------------------------------
# Simulate gene-by-gene mimicking real data
# --------------------------------------------------------------------------------


def simulate_gene(gene_id, id_array):
    gene_class_counts: CompatClassesDf = get_gene_class_counts(
        gene_id,
        tx_annot,
        all_compatibility_classes,
    )
    gene_effective_lengths = effective_lengths.filter(gene_id=gene_id).select(
        "mouse_id", "haplotype", "effective_length"
    )

    # Cleaner gene data for gene-level values like q_hat
    gene_data = prepare_gene_data(
        gene_id,
        gene_class_counts,
        genotypes,
        gene_effective_lengths,
        config=dict(
            min_genotype_confidence=0.9,
            drop_all_zeros=True,
        ),
    )
    # Gene data for all samples
    gene_data_all = prepare_gene_data(
        gene_id,
        gene_class_counts,
        genotypes,
        gene_effective_lengths,
        config=dict(
            min_genotype_confidence=0.0,
            drop_all_zeros=False,
        ),
    )

    assert gene_data_all.class_names == gene_data.class_names

    if id_array is None:
        id_array = np.array(gene_data_all.ids)
    assert (id_array == gene_data_all.ids).all()

    # ------------------------------
    # Gene-level parameters
    # ------------------------------

    if rng.random() > config["sigma_u_zero_frac"]:
        sigma_u = np.exp(
            rng.normal(config["log_sigma_u_mean"], config["log_sigma_u_std"])
        )
    else:
        sigma_u = 0

    if rng.random() > config["sigma_gamma_zero_frac"]:
        sigma_gamma = np.exp(
            rng.normal(config["log_sigma_gamma_mean"], config["log_sigma_gamma_std"])
        )
    else:
        sigma_gamma = 0

    if rng.random() > config["beta_frac_zero"]:
        has_ase = True
        beta = rng.normal(0, config["beta_std"], size=n_haps)
        beta = beta - beta.mean()  # sums to zero
    else:
        has_ase = False
        beta = np.zeros(n_haps)

    # Gammas are orthogonal beta, iid normally distributed
    Gm = scipy.linalg.null_space(gene_data.Bm.T)
    gamma = Gm @ rng.normal(0, sigma_gamma, size=Gm.shape[1])

    # Relative expression of each haplotype pair A vs B, ...
    delta = gene_data.Bm @ beta + gamma

    # Class proportions by hap: hap x class
    # TODO: consider pulling these from the ASE model fit values instead
    q = gene_data.q_hat

    gene_base_expr = gene_data_all.counts.sum(axis=1).mean(axis=0)

    # Buffering factor
    has_buffering = has_ase & (rng.random() > config["buffering_frac"])
    if has_buffering:
        buffering_factor = rng.uniform(
            config["buffering_factor_lo"], config["buffering_factor_hi"]
        )
    else:
        buffering_factor = 1

    # Covariate effects
    sex_effect = rng.normal(0, config["sex_effect_std"])
    DO_wave_effects = rng.normal(0, config["DO_wave_effect_std"], size=n_DO_waves)

    # ------------------------------
    # Sample-level parameters
    # ------------------------------

    ids_df = pl.DataFrame({"mouse_id": gene_data_all.ids})

    # Sample-level relative expression of hap1 vs hap2
    # 0 for homozygotes
    n_samples, n_classes = gene_data_all.counts.shape
    u = rng.normal(0, sigma_u, size=n_samples)

    # sample x class dispersion value
    alpha = gene_data_all.alpha

    # Covariates
    pheno = all_pheno.join(ids_df, "mouse_id", how="right", maintain_order="right")
    M = formulaic.model_matrix(
        "~ 0 + DOwave + sex",
        data=pheno,
        na_action="raise",
    )
    assert M.columns == [
        "DOwave[1]",
        "DOwave[2]",
        "DOwave[3]",
        "DOwave[4]",
        "DOwave[5]",
        "sex[T.M]",
    ]
    covariate_effects = M @ np.concatenate([DO_wave_effects, [sex_effect]])

    # sample offset terms
    sf = size_factors.join(ids_df, "mouse_id", how="right", maintain_order="right")[
        "size_factor"
    ].to_numpy()
    eff_len1 = gene_data_all.eff_len1
    eff_len2 = gene_data_all.eff_len2
    mean_eff_len = (eff_len1.mean() + eff_len2.mean()) / 2

    # Sample allele total expression - before buffering
    beta1 = beta[gene_data_all.hap1]
    beta2 = beta[gene_data_all.hap2]

    lam = buffering_factor

    total_mean = (
        gene_base_expr
        / 2
        * (eff_len1 * np.exp(lam * beta1) + eff_len2 * np.exp(lam * beta2))
        / mean_eff_len
        * np.exp(covariate_effects)
        * sf
    )

    # allelic ratio exactly as the ASE model defines eta (includes log length ratio)
    eta = (
        gene_data_all.sign * delta[gene_data_all.idx] + u + np.log(eff_len1 / eff_len2)
    )
    p = 1 / (1 + np.exp(-eta))

    class_props = (
        p[:, None] * q[gene_data_all.hap1] + (1 - p)[:, None] * q[gene_data_all.hap2]
    )
    mu0 = total_mean[:, None] * class_props

    log_disp = float(total_fits.filter(gene_id=gene_id)["log_disp"].item())
    log_disp_sd = float(total_fits.filter(gene_id=gene_id)["log_disp_se"].item())
    phi = np.exp(
        log_disp + log_disp_sd**2 / 2
    )  # Total counts dispersion parameterized as var = mu + phi mu^2

    sigma_total, c_mean = sigma_total_for_target(mu0, alpha, phi)
    if sigma_total == 0.0 and c_mean > phi:
        print(
            f"{gene_id}: class-level dispersion (c={c_mean:.3g}) exceeds total-model phi={phi:.3g}"
        )

    # Per-sample log-normal mean 0 noise of total expression to match observed total diserpsion
    eps = rng.normal(-(sigma_total**2) / 2, sigma_total, size=n_samples)
    mu = mu0 * np.exp(eps)[:, None]

    # ------------------------------
    # Count data
    # ------------------------------

    counts = rng.negative_binomial(alpha, alpha / (mu + alpha))

    # ------------------------------
    # Gather values
    # ------------------------------
    gene_truth = {
        "gene_id": gene_id,
        "has_ase": has_ase,
        "has_buffering": has_buffering,
        "buffering_factor": buffering_factor,
        "beta": beta,
        "delta": delta,
        "gamma": gamma,
        "sigma_u": sigma_u,
        "sigma_gamma": sigma_gamma,
        "sigma_total": sigma_total,
        "phi": phi,
        "c_mean": c_mean,
        "sex_effect": sex_effect,
        "DO_wave_effects": DO_wave_effects,
    }
    sample_truth = {
        "gene_id": [gene_id] * n_samples,
        "mouse_id": gene_data_all.ids,
        "eta": eta,  # allelic-ratio including length effects
        "mu": mu,
        "alpha": alpha,
        "u": u,
        "p": p,
        "total_mean": total_mean,
        "eps": eps,
    }
    return gene_truth, sample_truth, counts, gene_id, gene_class_counts.compat, id_array


# ------------------------------
# Run the simulations
# ------------------------------

gene_ids = []
gene_truths = []
sample_truths = []
class_counts = []
compats = []
id_array = None

for gene_id in config["gene_ids"]:
    print(f"Simulating: {gene_id}", flush=True)
    try:
        gene_truth, sample_truth, counts, gene_id, compat, id_array = simulate_gene(
            gene_id, id_array
        )
    except Exception as e:
        print(f"\tFailed on {gene_id}:\n\t{e}")
        continue

    gene_truths.append(gene_truth)
    sample_truths.append(sample_truth)
    class_counts.append(counts)
    gene_ids.append(gene_id)
    compats.append(compat)

assert id_array is not None

# ------------------------------
# Output values
# ------------------------------

gene_truth = pl.DataFrame(gene_truths)
_cols = list(sample_truths[0].keys())
sample_truth = pl.DataFrame(sample_truths).explode(_cols, empty_as_null=False)

gene_truth.write_parquet(outdir / "gene_truths.parquet")
sample_truth.write_parquet(outdir / "sample_truths.parquet")

# ------------------------------
# Construct compressed .h5 Emase files per sample
# ------------------------------

emase_outdir = outdir / "gbrs"
emase_outdir.mkdir(exist_ok=True)
all_class_counts = np.concatenate(class_counts, axis=1)
lname = tx_annot["transcript_id"].to_numpy()
canonical_txs = tx_annot.filter(pl.col("tag").str.contains("Ensembl_canonical"))
tx_map = (
    canonical_txs.select("transcript_id", "gene_id")
    .join(
        pl.DataFrame({"gene_id": gene_ids}),
        "gene_id",
        maintain_order="right",
    )
    .join(
        pl.DataFrame({"transcript_id": lname, "idx": np.arange(len(lname))}),
        "transcript_id",
        maintain_order="left",
    )
)
all_compat = np.concatenate(compats, axis=0)
gene_ids_rep = np.concatenate(
    [[id] * len(compat) for id, compat in zip(gene_ids, compats)]
)
tx_idxs = (
    pl.DataFrame({"gene_id": gene_ids_rep})
    .join(tx_map, "gene_id", maintain_order="left")["idx"]
    .to_numpy()
)
n_classes = all_class_counts.shape[1]


def get_hap_compat(h):
    (is_compat,) = np.where(all_compat[:, h])
    i = tx_idxs[is_compat]
    j = np.arange(n_classes)[is_compat]
    ones = np.ones(len(is_compat))
    coo = scipy.sparse.coo_matrix((ones, (i, j)), shape=(len(lname), n_classes))
    return scipy.sparse.csr_matrix(coo)


hap_compats = {f"h{h}": get_hap_compat(h) for h in range(n_haps)}
for i, mouse_id in enumerate(id_array):
    emase = Emase(
        shape=(len(lname), n_haps, n_classes),
        count=all_class_counts[i, :],
        haps=hap_compats,
        lname=lname,
        hname=HAPLOTYPES,
    )
    write_compressed_emase(
        emase,
        outpath=str(emase_outdir / f"{mouse_id}.compressed.h5"),
    )

# ------------------------------
# Final sanity check: confirm that an output emase
# matches the expected simulated counts for one sample + one gene
# ------------------------------

mouse_idx = 0
mouse_id = id_array[mouse_idx]
new = load_compressed_emase(
    str(emase_outdir / f"{mouse_id}.compressed.h5"),
    haplotypes=[f"h{h}" for h in range(n_haps)],
)
for gene_idx, gene_id in enumerate(gene_ids):
    sim_output: CompatClassesDf = get_gene_class_counts(
        gene_id,
        tx_annot,
        {mouse_id: new},
    )
    assert np.all(sim_output.compat == compats[gene_idx])
    assert np.all(sim_output.counts == class_counts[gene_idx][mouse_idx])
