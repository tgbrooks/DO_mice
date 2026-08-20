"""
Simulate count-level data that follows our buffering model to see how well it works
"""

import pathlib
import polars as pl
import numpy as np
import scipy.stats

tissue = "simulated_counts"
try:
    SEED = snakemake.params.SEED
    N_SAMPLES = snakemake.params.N_SAMPLES
    N_NO_CIS_GENES = snakemake.params.N_NO_CIS_GENES
    N_NO_BUFFERING_GENES = snakemake.params.N_NO_BUFFERING_GENES
    N_BUFFERING_GENES = snakemake.params.N_BUFFERING_GENES
except NameError:
    # for testing
    SEED = 100
    N_SAMPLES = 200
    N_NO_CIS_GENES = 50
    N_NO_BUFFERING_GENES = 50
    N_BUFFERING_GENES = 50
N_GENES = N_NO_CIS_GENES + N_NO_BUFFERING_GENES + N_BUFFERING_GENES
HAPLOTYPES = np.array(list("ABCDEFGH"))
N_HAPLOTYPES = len(HAPLOTYPES)
MODELS = ["POISSON", "NEGATIVE_BINOMIAL", "SHARED_DISPERSION"]

rng = np.random.default_rng(seed=SEED)

##################################################
# Generate the true parameters for each gene
##################################################

models = rng.choice(MODELS, size=N_GENES)
mean_expr = 10 ** rng.normal(2, 1, size=N_GENES)  # per allele, so total is 2x this
dispersion = 10 ** rng.normal(-1, 0.5, size=N_GENES)
fraction_unique = rng.uniform(0.05, 0.3, size=N_GENES)
haplotype_effects = np.concatenate(
    [
        np.zeros((N_NO_CIS_GENES, N_HAPLOTYPES)),
        rng.normal(
            0, 0.5, size=((N_BUFFERING_GENES + N_NO_BUFFERING_GENES), N_HAPLOTYPES)
        ),
    ]
)
# Mean of exp(hap_effect) has to be 1 so that 'mean_expr' is the true mean expr per allele
haplotype_effects = (
    haplotype_effects - np.log(np.exp(haplotype_effects).mean(axis=1))[:, None]
)
buffering_effects = np.concatenate(
    [
        np.zeros((N_NO_CIS_GENES + N_NO_BUFFERING_GENES)),
        -rng.uniform(0, 1, size=N_BUFFERING_GENES),
    ]
)

##################################################
# Generate genotypes and compute sample-level parameters
##################################################

# Sample genotype effect
diplotypes = rng.choice(N_HAPLOTYPES, size=(N_SAMPLES, N_GENES, 2))
diplotype_strs = HAPLOTYPES[diplotypes[:, :, 0]] + HAPLOTYPES[diplotypes[:, :, 1]]
# 0,1,2 encoded genotypes
genotypes = np.zeros((N_SAMPLES, N_GENES, N_HAPLOTYPES))
idx = np.meshgrid(np.arange(N_SAMPLES), np.arange(N_GENES))
genotypes[*idx, diplotypes[:, :, 0].T] += 1
genotypes[*idx, diplotypes[:, :, 1].T] += 1

base_haplotype_means = np.exp(haplotype_effects) * mean_expr[:, None]
haplotype_means = base_haplotype_means * genotypes

# Apply buffering: an effect that moves us towards the original 2*mean_expr of the gene
# buffering_effect = -1 gives perfect buffering counteracting any change in the mean
# buffering_effect = 0 gives no buffering
total_means = haplotype_means.sum(axis=-1)
buffering_factor = (total_means / (2 * mean_expr))[:, :, None] ** (
    buffering_effects[:, None]
)
buffered_haplotype_means = haplotype_means * buffering_factor

##################################################
# Generate the actual read counts
# using one of three models
##################################################
allele_counts = np.zeros((N_SAMPLES, N_GENES, N_HAPLOTYPES), dtype=int)
# Poisson model:
allele_counts[:, models == "POISSON"] = rng.poisson(
    buffered_haplotype_means,
    size=(N_SAMPLES, N_GENES, N_HAPLOTYPES),
)[:, models == "POISSON"]
# Negative binomial: mu = r(1-p)/p, dispersion = 1/r
# so p = 1/(mu dispersion + 1)
p = 1 / (buffered_haplotype_means * dispersion[:, None] + 1)
r = 1 / dispersion[:, None]
allele_counts[:, models == "NEGATIVE_BINOMIAL"] = rng.negative_binomial(
    r,
    p,
    size=(N_SAMPLES, N_GENES, N_HAPLOTYPES),
)[:, models == "NEGATIVE_BINOMIAL"]
# Shared dispersion model: poisson-gamma where the gamma variance
# is shared between the two alleles
scale_factor = rng.gamma(  # a mean 1 gamma with varying variance
    1 / dispersion,
    dispersion,
)
allele_counts[:, models == "SHARED_DISPERSION"] = rng.poisson(
    buffered_haplotype_means * scale_factor[None, :, None],
    size=(N_SAMPLES, N_GENES, N_HAPLOTYPES),
)[:, models == "SHARED_DISPERSION"]

##################################################
# Binomial thinning to get the *unique* counts
##################################################
allele_unique_counts = rng.binomial(
    allele_counts,
    fraction_unique[None, :, None],
    size=(N_SAMPLES, N_GENES, N_HAPLOTYPES),
)
total_counts = allele_counts.sum(axis=-1)


##################################################
# Check that counts follow the expected ratios:
##################################################
def extract_by_diplotypes(arr, diplotypes):
    # for arr and n_samples x n_genes x n_haplotypes
    # extract the part that is n_samples x n_genes x 2
    # corresponding to its two diplotypes
    # For hets, returns the same value twice
    samples, genes = np.meshgrid(np.arange(N_SAMPLES), np.arange(N_GENES))
    hap1 = arr[
        samples.T,
        genes.T,
        diplotypes[:, :, 0],
    ]
    hap2 = arr[
        samples.T,
        genes.T,
        diplotypes[:, :, 1],
    ]
    return np.moveaxis(np.array([hap1, hap2]), [0, 1, 2], [2, 0, 1])


diplotype_ac = extract_by_diplotypes(allele_counts, diplotypes)
ac_ratios = diplotype_ac[:, :, 0] / diplotype_ac[:, :, 1]
diplotype_au = extract_by_diplotypes(allele_unique_counts, diplotypes)
au_ratios = diplotype_au[:, :, 0] / diplotype_au[:, :, 1]
diplotype_hap_effects = extract_by_diplotypes(haplotype_means, diplotypes)
expected_ratios = diplotype_hap_effects[:, :, 0] / diplotype_hap_effects[:, :, 1]
high_expr = mean_expr > 100


def associate(x, y):
    x = x.flatten()
    y = y.flatten()
    nonnan = np.isfinite(y)
    return scipy.stats.linregress(x[nonnan], y[nonnan])


r_ac = associate(expected_ratios[:, high_expr], ac_ratios[:, high_expr])
r_au = associate(expected_ratios[:, high_expr], au_ratios[:, high_expr])

##################################################
# Output files in the required manner
##################################################

outdir = pathlib.Path(f"processed/{tissue}")
outdir.mkdir(exist_ok=True)

mouse_ids = [f"SIM{i:03}" for i in range(N_SAMPLES)]
mouse_ids_grid = np.array([[m] * N_GENES for m in mouse_ids])
gene_ids = [f"GENE{i:04}" for i in range(N_GENES)]
gene_ids_grid = np.array([gene_ids for _ in mouse_ids])
counts = pl.DataFrame(
    {
        "mouse_id": mouse_ids_grid.flatten(),
        "gene_id": gene_ids_grid.flatten(),
        "total": total_counts.flatten(),
    }
)
counts.write_parquet(
    f"processed/{tissue}/{tissue}.diploid.genes.founder_expected_read_counts.parquet"
)

au_dir = outdir / "gbrs_allele_unique_reads"
au_dir.mkdir(exist_ok=True)
for i, mouse_id in enumerate(mouse_ids):
    au_file = f"processed/{tissue}/gbrs_allele_unique_reads/{mouse_id}.allele_unique_reads.parquet"
    au = pl.DataFrame(
        {
            "gene_id": gene_ids,
            "total_reads": total_counts[i],
            "allele_specific_reads": allele_unique_counts[i].sum(axis=-1),
            "diplotype": diplotype_strs[i],
            "haplotype_1_unique": allele_unique_counts[
                i,
                np.arange(N_GENES),
                diplotypes[i, :, 0],
            ],
            "haplotype_2_unique": allele_unique_counts[
                i,
                np.arange(N_GENES),
                diplotypes[i, :, 1],
            ],
            "diplotype_incompat_reads": np.zeros(N_GENES),
        }
    )
    au.write_parquet(au_file)

annot = pl.DataFrame(
    {
        "gene_id": gene_ids,
        "gene_name": gene_ids,
        "chrom": "1",
        "start": 0,  # don't matter for this analysis
        "end": 1000,
        "strand": "+",
    }
)
annot.write_csv(outdir / "gene_annot.txt", separator="\t")
size_factors = pl.DataFrame(
    {"mouse_id": mouse_ids, "size_factor": 1}
)  # No size variation here
size_factors.write_csv(outdir / "size_factors.txt", separator="\t")

# No kinship relation, despite what the genotypes say
kinship = np.eye(N_SAMPLES) / 2
kinship = pl.DataFrame(
    {
        "mouse_id": mouse_ids,
        **{mouse_id: kinship[:, i] for i, mouse_id in enumerate(mouse_ids)},
    }
)
kinship.write_csv(outdir / "kinship.txt", separator="\t")

genotypes_df = pl.DataFrame(
    {
        "gene_id": gene_ids_grid.flatten(),
        "mouse_id": mouse_ids_grid.flatten(),
        "genotype": diplotype_strs.flatten(),
    }
)
genotypes_df.write_parquet(outdir / "genotypes.parquet")

phenotypes = pl.DataFrame(
    {
        "mouse.id": mouse_ids,
        "sex": rng.choice(["M", "F"], size=N_SAMPLES),
        "DOwave": rng.choice(np.arange(1, 6), size=N_SAMPLES),
    }
)
phenotypes.write_csv(outdir / "phenotypes.csv")

true_params = pl.DataFrame(
    {
        "gene_id": gene_ids,
        "model": models,
        "mean_expr": mean_expr,
        "dispersion": dispersion,
        "buffering_effect": buffering_effects,
        "faction_unique": fraction_unique,
        **{
            f"effect_{hap}": haplotype_effects[:, i] for i, hap in enumerate(HAPLOTYPES)
        },
        "type": (
            ["no_cis"] * N_NO_CIS_GENES
            + ["no_buffering"] * N_NO_BUFFERING_GENES
            + ["buffering"] * N_BUFFERING_GENES
        ),
    }
)
true_params.write_csv(outdir / "true_params.txt", separator="\t")
