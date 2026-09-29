"""
Estimate the fragment length distribution of one paired-end sample.

Fragments are measured on transcript coordinates, using the per-end bowtie
alignments to the pooled founder transcriptome. Only read pairs that can be
placed unambiguously are used:

  * the transcript is the only isoform of its gene (in the reference GTF), so
    the fragment's length on the transcript is its length on the molecule;
  * the transcript is at least --min-transcript-length long in the founder it
    is measured on, so that long fragments are not selected against;
  * the gene is called homozygous in this mouse with a confidence of at least
    --min-genotype-confidence, and the pair is measured on that founder's copy
    of the transcript (other founders' copies can differ by indels);
  * each mate aligns exactly once to that copy, and neither mate aligns to any
    other gene;
  * the mates are on opposite strands, facing each other.

The fragment length is then the span from the leftmost to the rightmost aligned
base of the pair. Fragments shorter than the read length (adapter read-through)
don't align end-to-end, so they are missing from the distribution.

Transcripts are named `{transcript_id}_{haplotype}` in the BAMs, as in the
pooled transcriptome.
"""

import argparse
import json

import numpy as np
import polars as pl
import polars_bio as pb

parser = argparse.ArgumentParser("Estimate a sample's fragment length distribution")
parser.add_argument("--R1", required=True, help="R1 bam")
parser.add_argument("--R2", required=True, help="R2 bam")
parser.add_argument("--gtf", required=True, help="reference GTF of the pooled transcriptome")
parser.add_argument(
    "--genotypes",
    help="per-gene genotype confidence TSV from genoprobs_to_gbrs.py",
)
parser.add_argument(
    "--homozygous-for",
    help="treat every gene as homozygous for this founder instead of reading "
    "--genotypes (for simulated samples)",
)
parser.add_argument("--min-transcript-length", type=int, default=2000)
parser.add_argument("--min-genotype-confidence", type=float, default=0.99)
parser.add_argument("--out", required=True, help="output JSON")
args = parser.parse_args()
if (args.genotypes is None) == (args.homozygous_for is None):
    parser.error("give exactly one of --genotypes and --homozygous-for")

REVERSE = 0x10

# Single-isoform genes of the reference annotation
transcripts = (
    pb.scan_gtf(args.gtf, attr_fields=["gene_id", "transcript_id"])
    .filter(pl.col("type") == "transcript")
    .collect()
    .select("gene_id", "transcript_id")
    .unique()
)
single_isoform = transcripts.filter(pl.len().over("gene_id") == 1)

# Founder each gene is homozygous for in this mouse
if args.homozygous_for:
    homozygous = single_isoform.select("gene_id", hap=pl.lit(args.homozygous_for))
else:
    homozygous = (
        pl.read_csv(args.genotypes, separator="\t")
        .filter(
            pl.col("diplotype").str.slice(0, 1) == pl.col("diplotype").str.slice(1, 1),
            pl.col("confidence") >= args.min_genotype_confidence,
        )
        .select("gene_id", hap=pl.col("diplotype").str.slice(0, 1))
    )

# Transcript lengths per founder, from the BAM header
header = json.loads(pb.scan_bam(args.R1).config_meta.get_metadata()["source_header"])
tx_lengths = pl.from_dicts(json.loads(header["reference_sequences"])).select(
    chrom="name",
    transcript_id=pl.col("name").str.split("_").list.get(0),
    hap=pl.col("name").str.split("_").list.get(1),
    length="length",
)

eligible = (
    single_isoform.join(homozygous, "gene_id")
    .join(tx_lengths, ["transcript_id", "hap"])
    .filter(pl.col("length") >= args.min_transcript_length)
)
eligible_chroms = eligible["chrom"].implode()
print(f"{len(eligible)} eligible transcripts")


def scan_batches(path):
    return (
        pb.scan_bam(path, use_zero_based=True)
        .select("name", "chrom", "start", "end", "flags")
        .collect_batches()
    )


def eligible_alignments(path) -> pl.DataFrame:
    """Alignments of one end to eligible transcripts."""
    return pl.concat(
        batch.filter(pl.col("chrom").is_in(eligible_chroms))
        for batch in scan_batches(path)
    )


def reads_hitting_other_transcripts(path, names: pl.Series) -> pl.Series:
    """Of `names`, those with an alignment to a transcript other than the one
    their eligible alignment is on (any founder's copy of it is allowed)."""
    wanted = names.implode()
    hits = pl.concat(
        batch.filter(pl.col("name").is_in(wanted)).select(
            "name", transcript_id=pl.col("chrom").str.split("_").list.get(0)
        )
        for batch in scan_batches(path)
    )
    return (
        hits.group_by("name")
        .agg(pl.col("transcript_id").n_unique())
        .filter(pl.col("transcript_id") > 1)["name"]
    )


counts: dict[str, int] = {"eligible_transcripts": len(eligible)}

mates = []
for end, path in (("R1", args.R1), ("R2", args.R2)):
    alignments = eligible_alignments(path)
    # A mate aligning more than once to the transcript has no single position
    alignments = alignments.filter(pl.len().over("name", "chrom") == 1)
    mates.append(alignments)

pairs = mates[0].join(mates[1], ["name", "chrom"], suffix="_R2")
counts["pairs_on_eligible_transcripts"] = len(pairs)

multi_gene = pl.concat(
    [reads_hitting_other_transcripts(path, pairs["name"]) for path in (args.R1, args.R2)]
).unique()
pairs = pairs.filter(~pl.col("name").is_in(multi_gene.implode()))
counts["pairs_after_removing_other_genes"] = len(pairs)

pairs = pairs.with_columns(
    r1_reverse=(pl.col("flags") & REVERSE) != 0,
    r2_reverse=(pl.col("flags_R2") & REVERSE) != 0,
)
same_strand = pairs["r1_reverse"] == pairs["r2_reverse"]
counts["pairs_same_strand"] = int(same_strand.sum())
pairs = pairs.filter(~same_strand)

# Facing each other: the forward mate starts no later, and ends no later, than
# the reverse mate
pairs = pairs.with_columns(
    fwd_start=pl.when("r1_reverse").then("start_R2").otherwise("start"),
    fwd_end=pl.when("r1_reverse").then("end_R2").otherwise("end"),
    rev_start=pl.when("r1_reverse").then("start").otherwise("start_R2"),
    rev_end=pl.when("r1_reverse").then("end").otherwise("end_R2"),
)
inward = (pl.col("fwd_start") <= pl.col("rev_start")) & (
    pl.col("fwd_end") <= pl.col("rev_end")
)
counts["pairs_not_facing"] = int(pairs.select((~inward).sum()).item())
pairs = pairs.filter(inward)
counts["pairs_used"] = len(pairs)

r1_reverse_fraction = float(pairs["r1_reverse"].mean()) if len(pairs) else None
lengths = pairs.select(
    length=pl.max_horizontal("end", "end_R2") - pl.min_horizontal("start", "start_R2")
)["length"].to_numpy()

summary = {
    "counts": counts,
    # Fraction of pairs with R1 on the transcript's reverse strand: near 1 for
    # dUTP-stranded libraries, near 0.5 for unstranded ones
    "r1_reverse_fraction": r1_reverse_fraction,
}
if len(lengths):
    summary.update(
        mean=float(lengths.mean()),
        sd=float(lengths.std()),
        median=float(np.median(lengths)),
        quantiles={
            str(q): float(np.quantile(lengths, q))
            for q in (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)
        },
        # histogram[i] is the number of fragments of length i
        histogram=np.bincount(lengths).tolist(),
    )
for key, value in counts.items():
    print(f"{key}: {value}")

with open(args.out, "w") as f:
    json.dump(summary, f)
