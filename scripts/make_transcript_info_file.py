import polars as pl
import polars_bio as pb

gff = "gbrs_ref/v116/B.gff3.gz"
out = "gbrs_ref/v116/emase.fullTranscripts.info"

gff = snakemake.input.gff
out = snakemake.output.out

annot = (
    pb.scan_gff(gff, attr_fields=["Parent", "ID"])
    .filter(pl.col("ID").str.starts_with("transcript:"))
    .collect()
)

annot.select(
    transcript_id=pl.col("ID").str.strip_prefix("transcript:"),
    # the parts of emase we use don't use this column
    # and in fact the GBRS-paper Zenodo-provided file also just gives 0.0 for all transcripts
    length=pl.lit(0.0),
).sort(
    "transcript_id",
).write_csv(
    out,
    separator="\t",
    include_header=False,
)
