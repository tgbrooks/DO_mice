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

The two BAMs are streamed together (see util/paired_bam.py), so memory use does
not grow with their size.

Transcripts are named `{transcript_id}_{haplotype}` in the BAMs, as in the
pooled transcriptome.
"""

import argparse
import json
from collections import Counter

import numpy as np
import polars as pl
import polars_bio as pb
import pysam

from util.paired_bam import PairingStats, paired_reads

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
parser.add_argument(
    "--max-reads",
    type=int,
    help="stop after this many reads of either BAM (default: read everything)",
)
parser.add_argument(
    "--pairing-window",
    type=int,
    default=1_000_000,
    help="give up on finding a read's other end once its own BAM is this many "
    "reads further on",
)
parser.add_argument("--out", required=True, help="output JSON")
args = parser.parse_args()
if (args.genotypes is None) == (args.homozygous_for is None):
    parser.error("give exactly one of --genotypes and --homozygous-for")

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

r1_bam = pysam.AlignmentFile(args.R1, "rb", check_sq=False)
r2_bam = pysam.AlignmentFile(args.R2, "rb", check_sq=False)
if r1_bam.references != r2_bam.references:
    raise SystemExit("R1 and R2 BAMs have different reference sequences")

# Transcript copies (BAM reference IDs) per founder, with their lengths
references = pl.DataFrame(
    {"chrom": r1_bam.references, "length": r1_bam.lengths}
).with_row_index("ref_id").with_columns(
    transcript_id=pl.col("chrom").str.split("_").list.get(0),
    hap=pl.col("chrom").str.split("_").list.get(1),
)
eligible = set(
    single_isoform.join(homozygous, "gene_id")
    .join(references, ["transcript_id", "hap"])
    .filter(pl.col("length") >= args.min_transcript_length)["ref_id"]
)
print(f"{len(eligible)} eligible transcripts")

# Transcript of each reference ID, as an integer, so that copies of the same
# transcript in different founders compare equal
transcript_of = (
    references.select(pl.col("transcript_id").rank("dense"))["transcript_id"].to_list()
)


def summarize(alignments):
    """Reduce one end's alignments to what pairing needs.

    None if the end has no alignment to an eligible transcript copy; otherwise
    (reference ID, start, end, is reverse, number of alignments to that copy,
    whether it also aligns to another transcript).
    """
    hits = [a for a in alignments if a.reference_id in eligible]
    if not hits:
        return None
    ref_id = hits[0].reference_id
    transcript = transcript_of[ref_id]
    on_copy = [a for a in hits if a.reference_id == ref_id]
    other_transcript = any(
        transcript_of[a.reference_id] != transcript
        for a in alignments
        if not a.is_unmapped
    )
    first = on_copy[0]
    return (
        ref_id,
        first.reference_start,
        first.reference_end,
        first.is_reverse,
        len(on_copy),
        other_transcript,
    )


counts = Counter()
lengths = Counter()
r1_reverse = 0
pairing = PairingStats()

for _, r1, r2 in paired_reads(
    r1_bam,
    r2_bam,
    summarize,
    window=args.pairing_window,
    max_reads=args.max_reads,
    stats=pairing,
):
    if r1 is None or r2 is None or r1[0] != r2[0]:
        continue
    counts["pairs_on_eligible_transcripts"] += 1
    ref_id, start1, end1, rev1, n1, other1 = r1
    _, start2, end2, rev2, n2, other2 = r2
    if n1 > 1 or n2 > 1:
        counts["pairs_aligned_more_than_once"] += 1
        continue
    if other1 or other2:
        counts["pairs_on_other_genes"] += 1
        continue
    if rev1 == rev2:
        counts["pairs_same_strand"] += 1
        continue
    # Facing each other: the forward mate starts no later, and ends no later,
    # than the reverse mate
    (fwd_start, fwd_end), (rev_start, rev_end) = (
        ((start2, end2), (start1, end1)) if rev1 else ((start1, end1), (start2, end2))
    )
    if fwd_start > rev_start or fwd_end > rev_end:
        counts["pairs_not_facing"] += 1
        continue
    counts["pairs_used"] += 1
    r1_reverse += rev1
    lengths[max(end1, end2) - min(start1, start2)] += 1

counts["eligible_transcripts"] = len(eligible)
counts["reads_R1"], counts["reads_R2"] = pairing.reads
counts["read_pairs"] = pairing.pairs
counts["unpaired_reads"] = pairing.orphans
counts["max_pending_reads"] = pairing.max_pending

used = counts["pairs_used"]
summary = {
    "counts": dict(counts),
    # Fraction of pairs with R1 on the transcript's reverse strand: near 1 for
    # dUTP-stranded libraries, near 0.5 for unstranded ones
    "r1_reverse_fraction": r1_reverse / used if used else None,
}
if used:
    # histogram[i] is the number of fragments of length i
    histogram = np.zeros(max(lengths) + 1, dtype=np.int64)
    for length, n in lengths.items():
        histogram[length] = n
    values = np.arange(len(histogram))
    mean = float((values * histogram).sum() / used)
    cumulative = np.cumsum(histogram) / used

    def quantile(q: float) -> float:
        return float(np.searchsorted(cumulative, q))

    summary.update(
        mean=mean,
        sd=float(np.sqrt((histogram * (values - mean) ** 2).sum() / used)),
        median=quantile(0.5),
        quantiles={str(q): quantile(q) for q in (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)},
        histogram=histogram.tolist(),
    )
for key, value in counts.items():
    print(f"{key}: {value}")

with open(args.out, "w") as f:
    json.dump(summary, f)
