import polars as pl
import polars_bio as pb
import gzip
import re

gtf = (
    pb.scan_gtf("gbrs_ref/v116/B.gtf.gz", attr_fields=["gene_id", "transcript_id"])
    .filter(pl.col("type") == "transcript")
    .collect()
)

transcript_ids = set(
    pl.read_csv("gbrs_ref/v116/selected_transcripts.txt")["transcript_id"]
)
gene_ids = set(gtf.filter(pl.col("transcript_id").is_in(transcript_ids))["gene_id"])

gene_id_re = re.compile('gene_id "([a-zA-Z0-9]+)"')
transcript_id_re = re.compile('transcript_id "([a-zA-Z0-9]+)"')
lines_written = 0
with (
    gzip.open("gbrs_ref/v116/B.gtf.gz", "rt") as input,
    gzip.open("gbrs_ref/v116/reference.gtf.gz", "wt", compresslevel=6) as output,
):
    for i, line in enumerate(input):
        if line.startswith("#"):
            output.write(line)
            lines_written += 1
            continue
        # Output lines in our transcript/gene set
        # Note that some lines don't have transcript ids
        # so they are judged only based off the gene_id
        m = gene_id_re.search(line)
        if m:
            gene_id = m.groups()[0]
            if gene_id not in gene_ids:
                continue
            m = transcript_id_re.search(line)
            if m:
                transcript_id = m.groups()[0]
                if transcript_id not in transcript_ids:
                    continue

        output.write(line)
        lines_written += 1
print(f"Wrote {lines_written} lines")
