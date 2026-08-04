"""GBRS pipeline for Diversity Outbred (DO) mouse RNA-seq.

Downloads RNA-seq from the SRA (one GEO series per tissue, all from the same DO
cohort) and founder haplotype probabilities from Dryad, then runs GBRS
(Churchill lab) to quantify allele-specific expression against each mouse's own
diploid genome.

Run it from this directory:

    uv run snakemake --profile <profile> -j 20

See README.md for setup (containers, sample tables, GBRS reference files).
"""

configfile: "config.yaml"

# Sample tables, sample selection, and shared helpers:
include: "steps/common.smk"
# Raw data: SRA FASTQs and the GBRS reference bundle:
include: "steps/download.smk"
# Genotypes: genoprobs .RData -> per-gene diplotype calls:
include: "steps/genotypes.smk"
# GBRS: alignment, EMASE, quantification, (optional) genome reconstruction:
include: "steps/gbrs.smk"

# These are quick bookkeeping steps, not worth submitting as cluster jobs:
localrules: genotyped_mice, combine_tpm


def default_targets() -> list[str]:
    """Files built by `rule all`, for every tissue in the config `run` list."""
    targets = []
    for tissue in TISSUES:
        mice = MICE[tissue]
        if not mice:
            continue
        # Allele-specific expression against each mouse's own genome:
        targets += expand(
            "results/{tissue}/gbrs/{mouse}.diploid.genes.tpm", tissue=tissue, mouse=mice
        )
        targets += [
            f"results/{tissue}/{tissue}.diploid.genes.tpm.parquet",
            f"results/{tissue}/{tissue}.multiway.genes.tpm.parquet",
            f"results/{tissue}/{tissue}.diploid.genes.expected_read_counts.parquet",
            f"results/{tissue}/{tissue}.multiway.genes.expected_read_counts.parquet",
        ]
        targets += [f"processed/{tissue}/gbrs/{mouse}.bootstrap_quants.parquet" for mouse in mice]
        if GBRS["run_reconstruct"]:
            # GBRS's own genome reconstruction, to compare against the array
            # genotypes:
            targets += expand(
                "results/{tissue}/gbrs/{mouse}.genome.pdf", tissue=tissue, mouse=mice
            )
            targets += expand(
                "results/{tissue}/gbrs/{mouse}.interpolated.genoprobs.tsv",
                tissue=tissue,
                mouse=mice,
            )
    targets += [
            "results/genotypes.parquet",
    ]
    return targets


rule all:
    """Default targets: allele-specific expression for the selected mice."""
    input:
        default_targets(),
