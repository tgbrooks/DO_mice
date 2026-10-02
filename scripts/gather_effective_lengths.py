"""
Extract effective lengths from salmon output of all mice

Must account for Salmon's merging of identical features
"""

import polars as pl
import polars_bio as pb
import pathlib

tissue = "Adipose"
tissue = salmon.wildcards.tissue

geno = (
    pl.scan_parquet("processed/genotypes.parquet")
    .select(
        "mouse_id",
        "gene_id",
        haps=pl.col("genotype").str.split(""),
    )
    .collect()
)

tx_annot = (
    pb.scan_gtf(
        "gbrs_ref/v116/reference.gtf.gz", attr_fields=["gene_id", "transcript_id"]
    )
    .filter(pl.col("type") == "transcript")
    .collect()
)
HAPLOTYPES = list("ABCDEFGH")

# Salmon collapses some transcripts together if they have identical sequences.
# Map all quantified tx's to their equivalent transcripts, including themselves.
# Also maps salmon identifiers to transcript/gene/haplotype
tx_map = (
    pl.concat(
        [
            pl.read_csv(
                "gbrs_ref/v116/salmon_all_haps/duplicate_clusters.tsv", separator="\t"
            ),  # columns: RetainedRef, DuplicateRef
            *[
                tx_annot.select(
                    RetainedRef=pl.col("transcript_id") + f"_{hap}",
                    DuplicateRef=pl.col("transcript_id") + f"_{hap}",
                )
                for hap in HAPLOTYPES
            ],
        ]
    )
    .unique()
    .select(
        "RetainedRef",
        transcript_id=pl.col("DuplicateRef").str.split("_").list.get(0),
        haplotype=pl.col("DuplicateRef").str.split("_").list.get(1),
    )
    .join(
        tx_annot.select("gene_id", "transcript_id"),
        on="transcript_id",
    )
)

temp = []
for salmon_dir in pathlib.Path(f"processed/{tissue}/salmon/").glob("*"):
    if not (salmon_dir / "quant.sf").exists():
        continue
    mouse_id = salmon_dir.name

    quants = (
        pl.scan_csv(salmon_dir / "quant.sf", separator="\t")
        .select(
            "Name",
            mouse_id=pl.lit(mouse_id),
            effective_length="EffectiveLength",
        )
        .join(tx_map.lazy(), left_on="Name", right_on="RetainedRef")
        .select(
            "mouse_id",
            "gene_id",
            "transcript_id",
            "haplotype",
            effective_length="effective_length",
        )
        .join(
            geno.lazy(),
            ["mouse_id", "gene_id"],
            how="right",
        )
        .filter(pl.col("haplotype").is_in("haps"))
        .drop("haps")
        .collect()
    )

    temp.append(quants)

results = pl.concat(temp).select(
    pl.col("mouse_id").cast(pl.Categorical),
    pl.col("gene_id").cast(pl.Categorical),
    pl.col("transcript_id").cast(pl.Categorical),
    pl.col("haplotype").cast(pl.Enum(HAPLOTYPES)),
    "effective_length",
)
results.write_parquet(f"processed/{tissue}/effective_lengths.parquet")
