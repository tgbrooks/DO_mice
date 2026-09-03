import polars as pl
import polars_bio as pb
import pathlib

gtf_file = snakemake.input.gtf

annot = (
    pb.scan_gtf(gtf_file, attr_fields=["gene_id", "transcript_id"])
    .filter(pl.col("type") == "transcript")
    .collect()
)

# We only need the R1 reads since the R2 reads are exactly the same
fastq = pathlib.Path(snakemake.input.fastq)

print(f"Processing {fastq}")
haplotype = fastq.name.split("_")[0]
reads = (
    pb.scan_fastq(str(fastq))
    .select("name")
    .collect()
    .select(
        transcript_id=pl.col("name").str.split("_").list.get(0),
    )
)
tx_counts = (
    reads["transcript_id"]
    .value_counts(name="num_reads")
    .with_columns(haplotype=pl.lit(haplotype))
)

tx_counts.write_csv(snakemake.output.by_tx, separator="\t")

gene_counts = (
    tx_counts.join(
        annot.select("gene_id", "transcript_id"),
        "transcript_id",
    )
    .group_by(["gene_id", "haplotype"])
    .agg(num_reads=pl.col("num_reads").sum())
)

gene_counts.write_csv(snakemake.output.by_gene, separator="\t")
