import subprocess
import pathlib
import shutil
import h5py
import hashlib
import math
import pickle
import io
import numpy as np
import polars as pl
from scipy.sparse import csc_matrix, csr_matrix

tmp_dir = pathlib.Path(
    f"/tmp/gbrs_bootstrap_{snakemake.wildcards.tissue}_{snakemake.wildcards.mouse}"
)
tmp_dir.mkdir(exist_ok=True)
bootstrapped_h5_file = tmp_dir / "bootstrapped.h5"

# We make one copy of the file and modify it in-place to generate the bootstraps
# since most of the file stays the same (only count matrix needs to change)
shutil.copy(snakemake.input.h5, bootstrapped_h5_file)


def stable_seed(value):
    """Turns a list, tuple, string, or integer into a usable seed value for numpy RNG"""
    hasher = hashlib.blake2b()
    if isinstance(value, tuple | list):
        for v in value:
            h = stable_seed(v)
            nbytes = math.ceil(h.bit_length() / 8)
            hasher.update(h.to_bytes(nbytes))
    elif isinstance(value, str):
        hasher.update(value.encode())
    elif isinstance(value, int):
        nbytes = math.ceil(value.bit_length() / 8)
        hasher.update(value.to_bytes(nbytes))
    else:
        raise ValueError(f"Unrecognized type for: {value}")
    x = int.from_bytes(hasher.digest()) % (2**64)
    return x


rng = np.random.default_rng(
    seed=stable_seed((100, snakemake.wildcards.tissue, snakemake.wildcards.mouse))
)


def bootstrap(outfile, rng):
    f = h5py.File(snakemake.input.h5, "r")
    count = np.asarray(f["count"])
    n_reads = count.sum()
    shape = pickle.load(io.BytesIO(f["/"].attrs["shape"]))
    n_loci, n_hap, n_read_groups = shape
    new_count = rng.multinomial(n_reads, count / n_reads)

    out = h5py.File(outfile, "r+")
    nonzero = new_count != 0
    del out["count"]
    out["count"] = new_count[nonzero]
    new_n_readgroups = np.sum(nonzero)
    out["/"].attrs["shape"] = pickle.dumps(
        (n_loci, n_hap, new_n_readgroups), protocol=0
    )
    for hap in range(0, 8):
        hap = f"h{hap}"
        all_ones = np.ones(len(f[hap]["indices"]))
        orig = csr_matrix(
            (all_ones, f[hap]["indices"], f[hap]["indptr"]),
            shape=(n_loci, n_read_groups),
        )
        new = orig[:, nonzero]

        del out[hap]["indptr"]
        del out[hap]["indices"]
        out[hap]["indptr"] = new.indptr
        out[hap]["indices"] = new.indices
    out.close()


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
