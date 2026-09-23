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


class Stream:
    def __init__(self, path):
        self.batches = bam_batches(path)
        self.buf = None
        self.seen = set()  # current read, may continue into next batch
        self.completed = {}  # insertion-ordered set of fully buffered read names
        self.last = None
        self.done = False

    def knows(self, name):
        return name in self.completed or name in self.seen

    def pull(self):
        try:
            new = next(self.batches)
        except StopIteration:
            self.done = True
            self.completed.update(dict.fromkeys(self.seen))
            self.seen, self.last = set(), None
            return
        self.buf = new if self.buf is None else pl.concat([self.buf, new])
        if len(new):
            names = new["name"].unique(maintain_order=True).to_list()
            self.last = names[-1]
            self.completed.update(
                dict.fromkeys(n for n in [*self.seen, *names] if n != self.last)
            )
            self.seen = {self.last}


def evict(completed, finished):
    """Drop finished names plus any unfinished names that precede the last
    finished one in this stream's order: the other stream has already passed
    them, so they can never be matched."""
    cut = None
    for k in completed:
        if k in finished:
            cut = k
    orphans = set()
    if cut is None:
        return orphans
    for k in list(completed):
        del completed[k]
        if k not in finished:
            orphans.add(k)
        if k == cut:
            break
    return orphans


r1, r2 = Stream(R1_BAM), Stream(R2_BAM)
i = 0
processed_alignments = 0
n_orphans = 0
while not (r1.done and r2.done):
    r1_behind = r1.last is not None and r2.knows(r1.last)
    r2_behind = r2.last is not None and r1.knows(r2.last)
    if r1.done:
        r2.pull()
    elif r2.done:
        r1.pull()
    elif r1_behind and not r2_behind:
        r1.pull()
    elif r2_behind and not r1_behind:
        r2.pull()
    else:
        r1.pull()
        r2.pull()

    if r1.buf is None or r2.buf is None:
        continue

    finished = r1.completed.keys() & r2.completed.keys()
    ready = (
        r1.buf.join(r2.buf, ["name", "chrom"], how="inner", suffix="_R2")
        .filter(pl.col("name").is_in(finished))
        .unique(subset=["name", "chrom"])
    )

    orphans = evict(r1.completed, finished) | evict(r2.completed, finished)
    n_orphans += len(orphans)
    drop = finished | orphans
    r1.buf = r1.buf.filter(~pl.col("name").is_in(drop))
    r2.buf = r2.buf.filter(~pl.col("name").is_in(drop))

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
            f"Working sizes: {len(r1.completed)=} {len(r2.completed)=} {r1.buf.shape=} {r2.buf.shape=}"
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
