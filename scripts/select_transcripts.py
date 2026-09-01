import polars as pl
import polars_bio as pb

HAPLOTYPES = list("ABCDEFGH")
temp = []
for haplotype in HAPLOTYPES:
    gff = (
        pb.scan_gff(
            f"gbrs_ref/v116/{haplotype}.gff3.gz",
            attr_fields=[
                "ID",
                "Parent",
                "tag",
                "coverage",
                "sequence_ID",
                "extra_copy_number",
                "copy_num_ID",
            ],
        ).with_columns(haplotype=pl.lit(haplotype))
    ).collect()
    tx_lengths = (
        gff.filter(type="exon")
        .group_by("Parent")
        .agg(tx_length=(pl.col("end") - pl.col("start") + 1).sum())
    )
    transcripts = gff.filter(
        pl.col("ID").str.starts_with("transcript:ENSMUST")  # Just transcripts
    ).join(
        tx_lengths,
        left_on="ID",
        right_on="Parent",
    )
    print(f"Loaded {len(transcripts)} transcripts from {haplotype}")
    temp.append(transcripts)
gffs = pl.concat(temp)


# We didn't generate extra copies in the pipeline
assert not (gffs["extra_copy_number"] > 0).any()

all_haps = gffs.filter(pl.col("haplotype").n_unique().over("ID") == 8)
print(f"Identified {all_haps['ID'].n_unique()} transcripts in all haplotypes")

primary = all_haps.filter(pl.col("tag").str.contains("gencode_primary"))
print(f"Of those, {primary['ID'].n_unique()} were gencode primary")

len_diffs = primary.with_columns(
    len_diff=(
        pl.col("tx_length").max().over("ID") - pl.col("tx_length").min().over("ID")
    )
)
print(
    f"Of those, {len_diffs.filter(pl.col('len_diff') <= 10)['ID'].n_unique()} have all haplotypes within 10nt length"
)
print(
    f"Note: only {len_diffs.filter(len_diff=0)['ID'].n_unique()} have consistent lengths across all haplotypes"
)

selected = len_diffs.filter(pl.col("len_diff") <= 10)
print(f"Resulting in {selected['Parent'].n_unique()} genes")
IDs = selected.select(transcript_id=pl.col("ID").str.strip_prefix("transcript:"))
IDs.write_csv("gbrs_ref/v116/selected_transcripts.txt", separator="\t")
