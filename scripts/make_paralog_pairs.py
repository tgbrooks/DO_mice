"""
Build the ordered paralog pairs (and matched null pairs) for one tissue and split them
into chunks for the pipeline. See util/paralogs.py for the selection.

Writes into --outdir:
    pairs.parquet       all ordered pairs: pair_id, pair_type (paralog / null),
                        gene_a (locus), gene_b (response), and annotations
    report.txt          number of pairs left after each filter
    chunks/{i}.tsv      pair_id, pair_type, gene_a, gene_b; one model-fitting job each
"""

import argparse
import pathlib

import numpy as np
import polars as pl
import yaml

from util.paralogs import null_pairs, ordered, read_mouse_paralogs, select_pairs, unordered_pairs

parser = argparse.ArgumentParser()
parser.add_argument("--paralogs", help="Compara homologies TSV (mouse paralog rows suffice)")
parser.add_argument("--gene_annot", help="processed/gene_annot.txt")
parser.add_argument("--class_counts", help="processed/{tissue}/class_counts.parquet")
parser.add_argument("--outdir")
args = parser.parse_args()

config = yaml.safe_load(open("config.yaml"))
params = config["paralogs"]
chromosomes = [c for c in config["chromosomes"] if c not in ("X", "Y", "MT")]

paralogs = read_mouse_paralogs(args.paralogs)
gene_chrom = pl.read_csv(args.gene_annot, separator="\t", schema_overrides={"chrom": pl.String})

# Expression: median over mice of the reads unique to the gene (no other gene shares
# them), which are the reads the models use
class_counts = pl.read_parquet(args.class_counts, columns=["gene_id", "unique_count"])
expression = pl.DataFrame(
    {
        "gene_id": class_counts["gene_id"],
        "expression": [
            float(np.median(np.sum(np.asarray(u, dtype=float), axis=0)))
            for u in class_counts["unique_count"].to_list()
        ],
    }
)

pairs, report = select_pairs(
    paralogs,
    gene_chrom,
    expression,
    min_identity=params["min_identity"],
    max_family_size=params["max_family_size"],
    min_expression=params["min_median_unique_reads"],
    chromosomes=chromosomes,
)
paralog_ordered = ordered(pairs).with_columns(pair_type=pl.lit("paralog"))

candidates = (
    expression.filter(pl.col("expression") >= params["min_median_unique_reads"])
    .join(gene_chrom.select("gene_id", "chrom"), "gene_id")
    .filter(pl.col("chrom").is_in(chromosomes))
)
nulls = null_pairs(
    paralog_ordered,
    candidates,
    unordered_pairs(paralogs),
    n_per_pair=params["n_null_per_pair"],
    window=params["null_match_window"],
    seed=params["seed"],
).with_columns(pair_type=pl.lit("null"))

all_pairs = pl.concat([paralog_ordered, nulls], how="diagonal_relaxed").with_columns(
    pair_id=pl.concat_str("pair_type", "gene_a", "gene_b", separator="__")
)
assert all_pairs["pair_id"].is_unique().all()

outdir = pathlib.Path(args.outdir)
(outdir / "chunks").mkdir(parents=True, exist_ok=True)
all_pairs.write_parquet(outdir / "pairs.parquet")
size = params["pairs_per_chunk"]
cols = ["pair_id", "pair_type", "gene_a", "gene_b"]
for i, start in enumerate(range(0, all_pairs.height, size)):
    all_pairs.slice(start, size).select(cols).write_csv(outdir / "chunks" / f"{i}.tsv", separator="\t")

lines = [f"{n:>10,}  {label}" for label, n in report]
lines.append(f"{paralog_ordered.height:>10,}  ordered paralog pairs (both directions)")
lines.append(f"{nulls.height:>10,}  null pairs")
lines.append(f"{all_pairs.height:>10,}  pairs in total, in {len(range(0, all_pairs.height, size))} chunks")
(outdir / "report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
