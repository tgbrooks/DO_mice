# DO mouse GBRS pipeline

A Snakemake pipeline that quantifies allele-specific expression in Diversity
Outbred (DO) mice with [GBRS](https://github.com/churchill-lab/gbrs) (Churchill
lab), using the mice's own array-based genotypes rather than genomes
reconstructed from the RNA-seq.

The steps are:

1. Resolve a GEO series into a table of mice and SRA runs, and download the
   FASTQs from the SRA (`steps/download.smk`).
2. Download founder haplotype probabilities (`genoprobs`) from Dryad and convert
   them into the per-gene diplotype calls GBRS accepts (`steps/genotypes.smk`).
3. Align to the 8-way pooled transcriptome with bowtie, build EMASE matrices,
   and quantify expression per founder haplotype and against each mouse's own
   diploid genome (`steps/gbrs.smk`).

One tissue is one GEO series. Because the tissues come from the same DO cohort,
genotypes are downloaded and converted once per mouse (under `geno/`) and reused
by every tissue that mouse appears in; RNA-seq results are per tissue (under
`results/{tissue}/`). The adipose series `GSE266549` is configured to start
with; add more tissues under `tissues:` in `config.yaml`.

## Setup

### Prerequisites

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/), and
have Apptainer available to build and run containers. Snakemake is run through
`uv run` and its Python dependencies are in `pyproject.toml`.

### Containers

Every rule runs inside an Apptainer image, so run Snakemake with
`--use-singularity` (see `run_pipeline.sh`). Build the three images into
`images/`:

```shell
mkdir -p images
# Disable FIPS mode
echo 0 > /tmp/fips_off
# GBRS, EMASE, bowtie, samtools (the Churchill lab's own image)
apptainer build --bind /tmp/fips_off:/proc/sys/crypto/fips_enabled images/gbrs.sif containers/gbrs.def
# R, for reading the genoprobs .RData file
apptainer build --bind /tmp/fips_off:/proc/sys/crypto/fips_enabled images/rgeno.sif containers/rgeno.def
# SRA Toolkit, for downloading the FASTQs
apptainer build --ignore-subuid --ignore-fakeroot-command --bind /tmp/fips_off:/proc/sys/crypto/fips_enabled images/sratools.sif containers/sratools.def
# For R in more general contexts with more libraries installed
apptainer build --bind /tmp/fips_off:/proc/sys/crypto/fips_enabled images/rgeneral.sif containers/rgeneral.def
```

Bind any paths the jobs need with `--singularity-args "-B /your/path"`.

### GBRS reference files

GBRS needs precomputed reference data for the DO founder strains: the bowtie
index of the 8-way pooled transcriptome, the EMASE transcript metadata, gene
positions, and the genome grid. These are published as the GBRS supporting-files
bundle, [Zenodo 10.5281/zenodo.8289936](https://zenodo.org/records/8289936).

If the bundle is already unpacked somewhere (e.g. a shared copy on the cluster),
point `gbrs: data_dir:` in `config.yaml` at it and nothing is downloaded.
Otherwise set `gbrs: url:` to the bundle's `.tar.gz` (check the Zenodo record for
the current file name) and the `download_gbrs_reference` rule fetches and unpacks
it. The file names within the directory are all configurable, since they differ
between bundle versions; the defaults match the current release:

| Config key           | Default                                  | Used by                     |
| -------------------- | ---------------------------------------- | --------------------------- |
| `bowtie_index`       | `bowtie.transcriptome`                   | alignment                   |
| `transcript_info`    | `emase.fullTranscripts.info`             | `emase bam2emase`           |
| `gene2transcripts`   | `emase.gene2transcripts.tsv`             | `gbrs quantify`             |
| `transcript_lengths` | `emase.pooled.fullTranscripts.info`      | `gbrs quantify`             |
| `gene_pos`           | `ref.gene_pos.ordered_ensBuild_105.npz`  | genotype conversion         |
| `genome_grid`        | `ref.genome_grid.GRCm39.tsv`             | genotype conversion         |
| `emissions`          | `gbrs_emissions_all_tissues.avecs.npz`   | `gbrs reconstruct` (opt-in) |
| `transition_prob`    | `transition_probabilities/tranprob.DO.G{generation}.{sex}.npz` | `gbrs reconstruct` (opt-in) |

`gbrs reconstruct`, `interpolate`, and `plot` also read `ref.fa.fai` (and
`founder.hexcolor.info` for plotting) from the same directory via the
`GBRS_DATA` environment variable, which the rules set for you.

### Sample tables

Each tissue needs `samples/{tissue}.tsv`, which maps mouse IDs to SRA runs.
Generate it with `scripts/setup_geo_samples.py`, which needs internet access
(run it before Snakemake, since cluster nodes are often offline):

```shell
uv run scripts/setup_geo_samples.py --tissue Adipose --geo GSE266549 --srp SRP505574
```

It downloads the GEO series' SOFT file, pulls the mouse ID (e.g. `DO021`) out of
each sample title with the `mouse_id_pattern` regex, then asks the ENA portal API
for each experiment's runs and library layout. The result has one row per run:

```
mouse_id  geo_sample  experiment  run       layout  sex  generation
DO021     GSM8244001  SRX24455001 SRR28921001 PAIRED  F    23
```

`sex` and `generation` are optional; they are used only to pick the transition
probability file when the optional reconstruction step is enabled. If GEO is
unreachable, pass `--srp SRP...` to work from the SRA study alone (mouse IDs are
then taken from the ENA sample titles). Mice whose ID cannot be found are
reported and skipped.

A mouse with several runs has several rows; its FASTQs are concatenated before
alignment. Single- and paired-end datasets are both handled, but one mouse's
runs must all have the same layout.

### Genotypes

`genotypes: url:` points at the Dryad file of founder haplotype probabilities.
It is an `.RData` file holding a `genoprobs` list with one 3-D array per
chromosome (mice x 8 founders x markers), mice labelled like `DO021` as in GEO.
The pipeline:

1. downloads it to `geno/genoprobs.RData`;
2. exports the selected mice as `geno/alleleprobs/{mouse}.tsv.gz` with
   `scripts/export_genoprobs.R` (base R only — the array margins are identified
   from the dimnames, so the mice/founders/markers order does not have to match);
3. converts each mouse to `geno/gbrs_genotypes/{mouse}.genotypes.tsv` with
   `scripts/genoprobs_to_gbrs.py`, the `#Gene_ID<TAB>Diplotype` format that
   `gbrs quantify -G` expects.

The conversion assigns each gene the founder probabilities of its nearest
genotyped marker, converts them to founder dosages (summing to 2), and calls the
marker homozygous when the top founder reaches `hom_dosage_threshold` (1.5 by
default) or heterozygous between the top two founders otherwise. Genes are
matched to markers in whichever coordinate the GBRS gene position file uses (cM
or bp — detected automatically). Marker coordinates come from the GBRS genome
grid when the marker names match it, from the marker name itself when it encodes
a position (`1_3000000`), or from a map object in the `.RData` file; if none of
those work the run stops with an explanation rather than guessing. Calls are
restricted to the genes in the EMASE gene-to-transcript file, since `gbrs
quantify` errors out on a gene it doesn't know. Genes on chromosomes with no
genotype data (typically Y and MT) are left out, so they get zero expression in
the diploid quantification — the log says how many.

## Running

Limit how many mice are processed with `max_samples` in `config.yaml` (10 to
start with; set it to `null` for all of them). Mice are taken in sorted order
per tissue.

To also skip RNA-seq samples that have no genotype, generate the list of
genotyped mice first — it downloads the `.RData` file and writes
`geno/genotyped_mice.txt`, which sample selection reads on the next run:

```shell
uv run snakemake --profile <profile> genotyped_mice
```

Then run the pipeline (`-n` first for a dry run):

```shell
uv run snakemake --profile <profile> -j 20 -n
uv run snakemake --profile <profile> -j 20
```

or submit it with `./run_pipeline.sh`, which wraps the same command for LSF.

### Snakemake profile

A profile tells Snakemake how to submit jobs; put one in
`~/.config/snakemake/{name}/config.yaml` and pass it as `--profile {name}`. Rules
set their own `mem_mb`, `runtime`, and `threads`, which are passed on to the
scheduler. The settings this pipeline needs beyond the executor are:

```yaml
executor: slurm            # or lsf, or omit to run locally
default-resources:
  runtime: "4h"
  mem_mb: 8000
resources:
  sra_downloads: 3
use-singularity: true
latency-wait: 60
```

`sra_prefetch` consumes `sra_downloads=1`, so at most three runs download at
once. Extraction (`fasterq-dump`) is a separate rule with no such limit, because
it is slow but not a load on the SRA servers, and the prefetched `.sra` data is
`temp()` so it is deleted as soon as the FASTQs exist. Without a profile, pass
`--resources sra_downloads=3` on the command line.

### Outputs

For each mouse, under `results/{tissue}/gbrs/`:

- `{mouse}.multiway.genes.tpm` — expression apportioned across all 8 founders.
- `{mouse}.diploid.genes.tpm` — expression against the mouse's own genotype,
  with the called diplotype in the `notes` column. Isoform-level and expected
  read count files are written alongside.
- `{mouse}.compressed.h5` — the EMASE alignment matrix, kept so quantification
  can be re-run without re-aligning.

And per tissue, under `results/{tissue}/`:

- `{tissue}.diploid.genes.tpm.tsv` — gene x mouse table of total TPM.
- `{tissue}.diploid.genes.founder_tpm.tsv` — per-founder TPM in long form.

### Optional: GBRS genome reconstruction

Set `gbrs: run_reconstruct: true` to also run GBRS's own genome reconstruction
from the RNA-seq (`gbrs reconstruct` → `interpolate` → `plot` → `export`), which
adds a founder-mosaic PDF and a founder-dosage TSV per mouse. Comparing that
mosaic against the array genotypes is a good check for sample mixups. Setting
`gbrs: genotype_source: reconstructed` quantifies against those reconstructed
genotypes instead of the downloaded ones; this needs the emission and transition
probability files from the bundle, and the transition file must match each
mouse's DO generation and sex (`default_generation` / `default_sex`, overridden
per mouse by the sample table).

## Notes and caveats

- The Dryad genotype file, the Zenodo bundle, and GEO/SRA were not reachable from
  the environment this pipeline was written in, so the file-name and format
  assumptions above (bundle contents, `genoprobs` layout, GEO sample titles) are
  taken from the GBRS documentation and the dataset description. They are all
  configurable, and the scripts check their inputs and fail with a specific
  message rather than silently producing wrong calls — but expect to adjust a
  file name or two on the first run.
- `bowtie` reports every best-stratum alignment (`-a --best --strata -v 3`), as
  GBRS requires; the BAMs are large, so they are marked `temp()` and removed once
  converted to EMASE format.
- Paired-end reads are aligned one end at a time and paired afterwards by
  `emase get-common-alignments`, which is how GBRS handles pairing.
