"""
Summarize gene total and allele-unique reads for simulated reads from
each haplotype and each possible diplotype.
"""

import pathlib
import yaml
import polars as pl
import polars_bio as pb
from util.compressed_emase import load_compressed_emase
from util.summarize_count_types import summarize_count_types

cfg = yaml.load(pathlib.Path("config.yaml").open(), yaml.Loader)
haplotype_names = cfg["haplotypes"].split(",")

outfile = "temp.parquet"

# outfile = snakemake.output.gene_unique

annot = (
    pb.scan_gtf(
        cfg["gtf"],
        attr_fields=["gene_id", "transcript_id", "transcript_version"],
    )
    .filter(pl.col("type") == "transcript")
    .collect()
)

haplotypes = [f"h{i}" for i in range(8)]  # compressed.h5 labels them h0, ..., h7
results = []
for source_haplotype in haplotype_names:
    H5 = f"results/simulated_reads/gbrs/{source_haplotype}.compressed.h5"
    print(f"Processing {H5}")
    data = load_compressed_emase(H5, haplotypes)

    for other_hap in haplotype_names:
        print(f"Running {other_hap}")
        # Treat these reads as having come from each possible diplotype
        genotypes = annot.select(
            "gene_id",
            diplotype=pl.lit(f"{source_haplotype}{other_hap}"),
        ).unique()

        # Run the computations
        df = summarize_count_types(data, annot, genotypes)
        results.append(
            df.with_columns(
                source_haplotype=pl.lit(source_haplotype),
                other_haplotype=pl.lit(other_hap),
            )
        )
pl.concat(results).write_parquet(outfile)
