"""
Simulate count-level data that follows our buffering model to see how well it works
"""

import pathlib
import polars as pl
import numpy as np

tissue = "simulated_counts"
try:
    SEED = snakemake.params.SEED
    N_SAMPLES = snakemake.params.N_SAMPLES
    N_NO_CIS_GENES = snakemake.params.N_NO_CIS_GENES
    N_NO_BUFFERING_GENES = snakemake.params.N_NO_BUFFERING_GENES
    N_BUFFERING_GENES = snakemake.params.N_BUFFERING_GENES
except NameError:
    SEED = 100
    N_SAMPLES = 200
    N_NO_CIS_GENES = 100
    N_NO_BUFFERING_GENES = 100
    N_BUFFERING_GENES = 100
N_GENES = N_NO_CIS_GENES + N_NO_BUFFERING_GENES + N_BUFFERING_GENES
HAPLOTYPES = np.array(list("ABCDEFGH"))
N_HAPLOTYPES = len(HAPLOTYPES)

rng = np.random.default_rng(seed=SEED)

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
buffering_effects = np.concatenate(
    [
        np.zeros((N_NO_CIS_GENES + N_NO_BUFFERING_GENES)),
        -rng.uniform(0, 1, size=N_BUFFERING_GENES),
    ]
)

# Sample genotype effect
diplotypes = rng.choice(N_HAPLOTYPES, size=(N_SAMPLES, N_GENES, 2))
diplotype_strs = HAPLOTYPES[diplotypes[:, :, 0]] + HAPLOTYPES[diplotypes[:, :, 1]]
# 0,1,2 encoded genotypes
genotypes = np.zeros((N_SAMPLES, N_GENES, N_HAPLOTYPES))
idx = np.meshgrid(np.arange(N_SAMPLES), np.arange(N_GENES))
genotypes[*idx, diplotypes[:, :, 0].T] += 1
genotypes[*idx, diplotypes[:, :, 1].T] += 1

haplotype_means = np.exp(haplotype_effects) * mean_expr[:, None] * genotypes

# Apply buffering: an effect that moves us towards the original mean_expr of the gene
# buffering_effect = -1 gives perfect buffering counteracting any change in the mean
# buffering_effect = 0 gives no buffering
total_means = haplotype_means.sum(axis=-1)
buffered_haplotype_means = (
    haplotype_means
    * (total_means / mean_expr)[:, :, None] ** (buffering_effects[:, None])
)

# Generate the actual read counts
# negative binomial: mu = r(1-p)/p, dispersion = 1/r
# so p = 1/(mu dispersion + 1)
p = 1 / (buffered_haplotype_means * dispersion[:, None] + 1)
r = 1 / dispersion[:, None]
allele_counts = rng.negative_binomial(
    r,
    p,
    size=(N_SAMPLES, N_GENES, N_HAPLOTYPES),
)
allele_unique_counts = rng.binomial(
    allele_counts,
    fraction_unique[None, :, None],
    size=(N_SAMPLES, N_GENES, N_HAPLOTYPES),
)
total_counts = allele_counts.sum(axis=-1)

# Output files in the required manner
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
        "mean_expr": mean_expr,
        "dispersion": dispersion,
        "buffering_effects": buffering_effects,
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
