import pathlib
import yaml
import numpy as np
import polars as pl
import polars_bio as pb
from util.compressed_emase import load_compressed_emase
from util.summarize_count_types import summarize_count_types

cfg = yaml.load(pathlib.Path("config.yaml").open(), yaml.Loader)
haplotype_names = cfg["haplotypes"].split(",")

H5 = "processed/Adipose/gbrs/DO024.compressed.h5"
genotype_file = "geno/gbrs_genotypes/DO024.genotypes.tsv"
outfile = "temp.parquet"
gtf_file = "gbrs_ref/v116/reference.gtf.gz"
H5 = snakemake.input.h5
genotype_file = snakemake.input.genotypes
gtf_file = snakemake.input.gtf
outfile = snakemake.output.gene_unique

haplotypes = [f"h{i}" for i in range(8)]  # compressed.h5 labels them h0, ..., h7
data = load_compressed_emase(H5, haplotypes)

genotypes = pl.read_csv(genotype_file, separator="\t").rename(
    {"#Gene_ID": "gene_id", "Diplotype": "diplotype"}
)

annot = (
    pb.scan_gtf(
        gtf_file,
        attr_fields=["gene_id", "transcript_id", "transcript_version"],
    )
    .filter(pl.col("type") == "transcript")
    .collect()
)

transcript_genotypes = genotypes.join(
    annot.select("gene_id", "transcript_id"),
    "gene_id",
    how="right",
)

# Make sure the reference matches
assert not transcript_genotypes["transcript_id"].is_null().any(), (
    "all genotyped genes should be annotated"
)
assert np.isin(data.lname, annot["transcript_id"]).all(), (
    "all quantified transcripts should be annotated"
)
# NOTE: some quantified genes are not genotyped
# they will get masked out entirely

# Run the computations
df = summarize_count_types(data, annot, genotypes)
df.write_parquet(outfile)
