"""
Compute coverage per read alignment class as determined by EMASE.
For each class, store coverage of over all of its transcripts.
This way, we can see what each read class corresponds to positionally.

Reads are assumed to be paired end and stranded with R1 coming first.
BAM files must be grouped by reads with only the best alignments kept.

NOTE: if a read maps multiple times to the same transcript, we count
one R1 and one R2 alignment chosen arbitrarily.
"""

import polars as pl
import polars_bio as pb
import pyarrow
import pyarrow.parquet
import numpy as np
import itertools
import json
import argparse

from util.compressed_emase import load_compressed_emase

parser = argparse.ArgumentParser("Compute coverage per read alignment class")
parser.add_argument("--R1", help="R1 bam")
parser.add_argument("--R2", help="R2 bam")
parser.add_argument("--emase", help="compressed emase h5 file")
parser.add_argument("--out", help="output parquet")
args = parser.parse_args()
# R1_BAM = "processed/simulated_reads/gbrs/A.R1.bam"
# R2_BAM = "processed/simulated_reads/gbrs/A.R2.bam"
# COMPRESSED_EMASE = "processed/simulated_reads/gbrs/A.compressed.h5"
# OUTFILE = "temp_cov.parquet"
R1_BAM = args.R1
R2_BAM = args.R2
COMPRESSED_EMASE = args.emase
OUTFILE = args.out

cemase = load_compressed_emase(
    COMPRESSED_EMASE,
    haplotypes=[f"h{i}" for i in range(8)],
)
tx_id_to_num = {name.decode(): i for i, name in enumerate(cemase.lname)}


def get_class_map(cemase):
    """Represent a mapping from alignments to class numbers in the compressed emase file"""
    temp = []
    for hap_name, hap in zip(cemase.hname, cemase.haps.values()):
        row, col = hap.nonzero()
        temp.append(
            pl.DataFrame(
                {
                    "hap": hap_name,
                    "tx_num": row,
                    "class_num": col,
                }
            )
        )
    map = pl.concat(temp).with_columns(
        pl.col("hap").cast(pl.Enum(cemase.hname)),
        pl.col("tx_num").cast(pl.Int32),
    )
    return (
        map.sort("tx_num", "hap")
        .group_by("class_num")
        .agg(read_class=pl.struct("hap", "tx_num"))
    )


class_map = get_class_map(cemase)

# Extract transcript lengths from the BAM
metadata = pb.scan_bam(R1_BAM).config_meta.get_metadata()
metadata = json.loads(metadata["source_header"])
tx_lengths = (
    pl.from_dicts(json.loads(metadata["reference_sequences"]))
    .select(
        transcript_id=pl.col("name").str.split("_").list.get(0),
        hap=pl.col("name").str.split("_").list.get(1),
        length="length",
    )
    .with_columns(
        tx_num=pl.col("transcript_id").replace_strict(
            tx_id_to_num, return_dtype=pl.Int32
        ),
        hap=pl.col("hap").cast(pl.Enum(cemase.hname)),
    )
)

# Initialize zero coverage everywhere
coverage = {
    (read_class_num, hap, tx_num): np.zeros(tx_length, dtype=np.int32)
    for read_class_num, tx_num, hap, tx_length in class_map.explode(
        "read_class", empty_as_null=False
    )
    .unnest("read_class")
    .join(tx_lengths, ["hap", "tx_num"])
    .select("class_num", "tx_num", "hap", "length")
    .iter_rows()
}


def bam_batches(file):
    return (
        pb.scan_bam(file, use_zero_based=True)
        .select(
            "name",
            "chrom",
            "start",
            "end",
        )
        .collect_batches()
    )


R1 = bam_batches(R1_BAM)
R2 = bam_batches(R2_BAM)

R1_to_process = None
R2_to_process = None
R1_seen = set()
R2_seen = set()
R1_completed = set()
R2_completed = set()
R1_done = False
R2_done = False
i = 0
processed_alignments = 0
while not (R1_done and R2_done):
    try:
        R1_new = next(R1)
        if R1_to_process is None:
            R1_to_process = R1_new
        else:
            R1_to_process = pl.concat([R1_to_process, R1_new])
        if len(R1_new) > 0:
            R1_seen.update(R1_new["name"])
            R1_last_read = R1_new["name"][-1]  # Last read may continue in next batch
            new_completed = R1_seen.difference([R1_last_read])
            R1_completed.update(new_completed)
            R1_seen = {R1_last_read}
    except StopIteration:
        R1_done = True
        R1_completed.update(R1_seen)
        R1_seen = set()

    try:
        R2_new = next(R2)
        if R2_to_process is None:
            R2_to_process = R2_new
        else:
            R2_to_process = pl.concat([R2_to_process, R2_new])
        if len(R2_new) > 0:
            R2_seen.update(R2_new["name"])
            R2_last_read = R2_new["name"][-1]  # Last read may continue in next batch
            new_completed = R2_seen.difference([R2_last_read])
            R2_completed.update(new_completed)
            R2_seen = {R2_last_read}
    except StopIteration:
        R2_done = True
        R2_completed.update(R2_seen)
        R2_seen = set()

    assert R1_to_process is not None
    assert R2_to_process is not None

    finished = R1_completed & R2_completed
    ready = (
        R1_to_process.join(
            R2_to_process,
            ["name", "chrom"],  # 'chrom' includes tx and hap
            how="inner",
            suffix="_R2",
        )
        .filter(
            pl.col("name").is_in(finished),
        )
        .unique(subset=["name", "chrom"])
    )  # drop multiple alignments to the same transcript

    R1_to_process = R1_to_process.filter(~pl.col("name").is_in(finished))
    R2_to_process = R2_to_process.filter(~pl.col("name").is_in(finished))

    R1_completed.difference_update(finished)
    R2_completed.difference_update(finished)

    # Determine read classes
    ready = ready.with_columns(
        tx_num=pl.col("chrom")
        .str.split("_")
        .list.get(0)
        .replace_strict(tx_id_to_num)
        .cast(pl.Int32),
        hap=pl.col("chrom").str.split("_").list.get(1).cast(pl.Enum(cemase.hname)),
    ).sort("tx_num", "hap")

    agged = (
        ready.select("name", "hap", "tx_num")
        .unique(maintain_order=True)
        .group_by("name")
        .agg(read_class=pl.struct("hap", "tx_num"))
    )
    mapped = agged.join(class_map, "read_class", how="left")

    assert not mapped["class_num"].is_null().any()

    joined = ready.join(mapped.select("name", "class_num"), "name")

    # Compute coverage per class and transcript
    for tx_num, hap, class_num, start_R1, end_R1, start_R2, end_R2 in joined.select(
        "tx_num", "hap", "class_num", "start", "end", "start_R2", "end_R2"
    ).iter_rows():
        cov = coverage[(class_num, hap, tx_num)]
        if start_R2 < end_R1 and start_R1 < end_R2:
            # overlap -> count union once
            cov[min(start_R1, start_R2) : max(end_R1, end_R2)] += 1
        else:
            cov[start_R1:end_R1] += 1
            cov[start_R2:end_R2] += 1
        processed_alignments += 1
    i += 1
    if i % 100 == 0:
        print(f"Processed batches: {i}")
        print(
            f"Working sizes: {len(R1_completed)=} {len(R2_completed)=} {R1_to_process.shape=} {R2_to_process.shape=}"
        )

print(f"Processed {processed_alignments} alignments")

# Build a final parquet file of RLE coverage
print(f"Writing out to {OUTFILE}")
tx_num_to_id = {i: name.decode() for i, name in enumerate(cemase.lname)}
# Write out in batches
arrow_schema = pyarrow.schema(
    {
        "read_class": pyarrow.int32(),
        "hap": pyarrow.large_string(),
        "tx_num": pyarrow.int32(),
        "start": pyarrow.int32(),
        "len": pyarrow.int32(),
        "cov": pyarrow.int32(),
    }
)
j = 0
with pyarrow.parquet.ParquetWriter(OUTFILE, arrow_schema) as writer:
    for batch in itertools.batched(coverage.items(), n=1000):
        df = (
            pl.DataFrame(
                [
                    {
                        "read_class": read_class_num,
                        "hap": hap,
                        "tx_num": tx_num,
                        "cov": pl.Series(cov).rle(),
                    }
                    for (read_class_num, hap, tx_num), cov in batch
                ]
            )
            .select(
                pl.col("read_class").cast(pl.Int32),
                "hap",
                pl.col("tx_num").cast(pl.Int32),
                cov=pl.col("cov").list.eval(
                    pl.struct(
                        start=(
                            pl.element().struct.field("len").cum_sum()
                            - pl.element().struct.field("len")
                        ).cast(pl.Int32),
                        len=pl.element().struct.field("len").cast(pl.Int32),
                        cov=pl.element().struct.field("value").cast(pl.Int32),
                    )
                ),
            )
            .explode("cov", empty_as_null=False)
            .unnest("cov")
        )
        writer.write_table(df.to_arrow())
        j += 1
        if j % 100 == 0:
            print(f"Wrote {j} batches out")
