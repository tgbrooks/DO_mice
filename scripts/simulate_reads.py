import polars as pl
import polars_bio as pb
import gzip

haplotype = "B"
R1_out = "temp_1.fastq.gz"
R2_out = "temp_2.fastq.gz"

haplotype = snakemake.wildcards.haplotype
R1_out = snakemake.output.R1
R2_out = snakemake.output.R2

READ_LENGTH = 101
FRAGMENT_LENGTHS = [150, 200, 250, 300, 350, 400]
SKIP_POSITIONS = 10
fasta = (
    pb.scan_fasta("gbrs_ref/v116/all_haps.cdna.fa")
    .filter(pl.col("name").str.ends_with(haplotype))
    .collect()
).with_columns(
    transcript_id=pl.col("name").str.split("_").list.get(0),
)

# Find just the 'basic' transcripts
basic_annot = (
    pb.scan_gtf(
        "/project/itmatlab/genomes/mouse/GRCm39/Ensembl.v105/Mus_musculus.GRCm39.105.gtf.gz",
        attr_fields=["gene_id", "transcript_id", "tag"],
    )
    .filter(
        pl.col("type") == "transcript",
    )
    .collect()
    .filter(pl.col("tag").str.contains("basic"))
)

basic_fasta = fasta.join(basic_annot, "transcript_id", how="inner").select(
    "transcript_id", "sequence"
)


tab = str.maketrans("ATCG", "TAGC")


def rev_complement(seq):
    return seq.translate(tab)[::-1]


quals = "F" * READ_LENGTH + "\n"
with (
    gzip.open(R1_out, "wt", compresslevel=6) as R1,
    gzip.open(R2_out, "wt", compresslevel=6) as R2,
):
    for transcript, seq in basic_fasta.iter_rows():
        for start in range(0, len(seq), SKIP_POSITIONS):
            for frag_len in FRAGMENT_LENGTHS:
                end = start + frag_len
                if end >= len(seq):
                    continue
                frag_seq = seq[start:end]
                frag_fwd = rev_complement(frag_seq[:READ_LENGTH])
                frag_rev = frag_seq[-READ_LENGTH:]

                R1.write(f"@{transcript}_{start}_{end}\n")
                R1.write(frag_fwd + "\n")
                R1.write("+\n")
                R1.write(quals)

                R2.write(f"@{transcript}_{start}_{end}\n")
                R2.write(frag_rev + "\n")
                R2.write("+\n")
                R2.write(quals)
