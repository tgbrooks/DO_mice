import subprocess
import pathlib
import shutil
import numpy as np
import polars as pl
from util.random import stable_seed
from util.compressed_emase import load_compressed_emase, write_compressed_emase

tmp_dir = pathlib.Path(
    f"/tmp/gbrs_bootstrap_{snakemake.wildcards.tissue}_{snakemake.wildcards.mouse}"
)
tmp_dir.mkdir(exist_ok=True)
bootstrapped_h5_file = tmp_dir / "bootstrapped.h5"

# We make one copy of the file and modify it in-place to generate the bootstraps
# since most of the file stays the same (only count matrix needs to change)
shutil.copy(snakemake.input.h5, bootstrapped_h5_file)

rng = np.random.default_rng(
    seed=stable_seed((100, snakemake.wildcards.tissue, snakemake.wildcards.mouse))
)

HAPLOTYPES = [f"h{i}" for i in range(8)]


def bootstrap(outfile, rng):
    data = load_compressed_emase(snakemake.input.h5, haplotypes=HAPLOTYPES)
    n_reads = data.count.sum()
    n_loci, n_hap, n_read_groups = data.shape

    # Perform bootstrapping
    new_count = rng.multinomial(n_reads, data.count / n_reads)

    # Drop regions where we now have 0 reads: these cause problems downstream
    nonzero = new_count != 0
    new_n_readgroups = np.sum(nonzero)
    data.shape = (n_loci, n_hap, new_n_readgroups)
    for hap in HAPLOTYPES:
        data.haps[hap] = data.haps[hap][:, nonzero]
    write_compressed_emase(data, outfile)


results = []
for i in range(snakemake.params.n_bootstraps):
    # write a new bootstrapped count file
    bootstrap(bootstrapped_h5_file, rng)

    cmd = f"apptainer exec images/gbrs.sif gbrs quantify \
                -i {bootstrapped_h5_file} \
                -g {snakemake.input.gene2transcripts} \
                -L {snakemake.input.lengths} \
                -G {snakemake.input.genotypes} \
                -M {snakemake.params.model} \
                -a \
                -o {tmp_dir}/bootstrapped"
    subprocess.run(cmd, shell=True, check=True)

    counts_file = tmp_dir / "bootstrapped.diploid.genes.expected_read_counts"
    results.append(
        pl.read_csv(counts_file, separator="\t")
        .with_columns(bootstrap=pl.lit(i))
        .drop("notes")
    )

results = pl.concat(results)
results.write_parquet(snakemake.output.counts)
