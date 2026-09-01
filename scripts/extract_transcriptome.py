import polars as pl
import polars_bio as pb

haplotype = snakemake.wildcards.haplotype
print(f"Processing {haplotype}")
transcript_ids = pl.read_csv("gbrs_ref/v116/selected_transcripts.txt", separator="\t")

gff = pb.read_gff(f"gbrs_ref/v116/{haplotype}.gff3.gz", attr_fields=["Parent", "ID"])

fasta = pb.read_fasta(f"gbrs_ref/v116/{haplotype}.dna.fa.gz")

tab = str.maketrans("ATCG", "TAGC")


def rev_complement(seq):
    return seq.translate(tab)[::-1]


results = []
for transcript in transcript_ids["transcript_id"]:
    print(f"\t{transcript}")
    exons = gff.filter(Parent=f"transcript:{transcript}", type="exon")
    if len(exons) == 0:
        raise ValueError(f"No exons available for {transcript}")
    seqs = []
    for chrom, start, end, strand in exons.select(
        "chrom", "start", "end", "strand"
    ).iter_rows():
        chr_seq = fasta.filter(name=chrom)["sequence"][0]
        seqs.append(chr_seq[start - 1 : end])
    seq = "".join(seqs)
    if strand == "-":
        seq = rev_complement(seq)
    results.append({"transcript_id": transcript, "sequence": seq})
transcriptome = pl.DataFrame(results)

pb.write_fasta(
    transcriptome.select(
        name="transcript_id",
        description=pl.lit(""),
        sequence="sequence",
    ),
    f"gbrs_ref/v116/{haplotype}.cdna.fa.gz",
)
