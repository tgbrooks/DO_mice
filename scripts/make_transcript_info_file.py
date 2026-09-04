import polars as pl
import polars_bio as pb

gtf = "gbrs_ref/v116/reference.gtf.gz"
out = "gbrs_ref/v116/emase.fullTranscripts.info"

gtf = snakemake.input.gtf
out = snakemake.output.out

annot = (
    pb.scan_gtf(gtf, attr_fields=["transcript_id"])
    .filter(pl.col("type") == "transcript")
    .collect()
)

annot.select(
    transcript_id="transcript_id",
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
