"""
Fit the paralog compensation model (ase_buffering/paralog_model.py) to each ordered
pair of a chunk and save the posterior samples to one .nc file (util/netcdf.py), one
group per pair_id. Pairs that fail get an error group instead.
"""

import argparse
import json
import time

import numpy as np
import polars as pl
import pymc as pm
import yaml

from util.netcdf import GeneError, save_results
from ase_buffering.paralog_model import DEFAULTS, make_paralog_model

parser = argparse.ArgumentParser("Fit the paralog compensation model to a chunk of pairs")
parser.add_argument("--tissue")
parser.add_argument("--chunk", help="chunk TSV: pair_id, pair_type, gene_a, gene_b")
parser.add_argument("--class_counts", help="processed/{tissue}/class_counts.parquet")
parser.add_argument("--outfile", help=".nc output file")
parser.add_argument("--cores", type=int, default=4)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--draws", type=int, default=500)
parser.add_argument("--config", default=None, help="JSON dict overriding DEFAULTS")
args = parser.parse_args()

CONFIG = {**DEFAULTS, **(json.loads(args.config) if args.config else {})}
MIN_GENOTYPE_CONFIDENCE = 0.9  # as for the ASE and total models
MIN_MICE = 50

config = yaml.safe_load(open("config.yaml"))
HAPLOTYPES = config["haplotypes"].split(",")
HAP_TO_NUM = {h: i for i, h in enumerate(HAPLOTYPES)}

chunk = pl.read_csv(args.chunk, separator="\t")
genes = set(chunk["gene_a"]) | set(chunk["gene_b"])

# Response: per mouse, the reads unique to gene B (compatible with no other gene)
class_counts = (
    pl.scan_parquet(args.class_counts)
    .filter(pl.col("gene_id").is_in(list(set(chunk["gene_b"]))))
    .select("gene_id", "mouse_ids", "unique_count")
    .collect()
)
unique_reads = {
    row["gene_id"]: pl.DataFrame(
        {
            "mouse_id": [str(m) for m in row["mouse_ids"]],
            "y": np.sum(np.asarray(row["unique_count"], dtype=np.int64), axis=0),
        }
    )
    for row in class_counts.to_dicts()
}

genotypes = (
    pl.read_parquet("processed/genotypes.parquet")
    .filter(
        pl.col("gene_id").is_in(list(genes)),
        pl.col("genotype_confidence") > MIN_GENOTYPE_CONFIDENCE,
    )
    .select("gene_id", "mouse_id", "genotype")
)
sample_info = (
    pl.read_csv("phenotypes.csv.gz")
    .select(mouse_id="mouse.id", sex="sex", wave=pl.col("DOwave").cast(str))
    .join(
        pl.read_csv(f"processed/{args.tissue}/size_factors.txt", separator="\t"),
        "mouse_id",
    )
)


def pair_data(gene_a: str, gene_b: str) -> pl.DataFrame:
    """Mice with B's unique reads, confident genotypes at both loci, and covariates."""
    if gene_b not in unique_reads:
        raise ValueError(f"No class counts for {gene_b}")
    geno = lambda gene, name: genotypes.filter(gene_id=gene).select(  # noqa: E731
        "mouse_id", pl.col("genotype").alias(name)
    )
    df = (
        unique_reads[gene_b]
        .join(geno(gene_b, "diplotype_b"), "mouse_id")
        .join(geno(gene_a, "diplotype_a"), "mouse_id")
        .join(sample_info, "mouse_id")
        .sort("mouse_id")
    )
    if df.height < MIN_MICE:
        raise ValueError(f"Only {df.height} mice with data for {gene_a} -> {gene_b}")
    return df


results = {}
quiet = chunk.height > 1
for pair in chunk.to_dicts():
    pair_id = pair["pair_id"]
    try:
        start = time.time()
        print(f"Modelling {pair_id}", flush=True)
        df = pair_data(pair["gene_a"], pair["gene_b"])

        def haps(col, k):
            return np.array([HAP_TO_NUM[d[k]] for d in df[col]])

        waves = sorted(df["wave"].unique())
        with make_paralog_model(
            y=df["y"].to_numpy(),
            hap_b1=haps("diplotype_b", 0),
            hap_b2=haps("diplotype_b", 1),
            hap_a1=haps("diplotype_a", 0),
            hap_a2=haps("diplotype_a", 1),
            male=(df["sex"] == "M").cast(int).to_numpy(),
            wave=np.array([waves.index(w) for w in df["wave"]]),
            size_factor=df["size_factor"].to_numpy(),
            haplotypes=HAPLOTYPES,
            waves=waves,
            mouse_ids=df["mouse_id"].to_list(),
            config=CONFIG,
        ):
            idata = pm.sample(
                args.draws,
                cores=args.cores,
                chains=args.chains,
                random_seed=102,
                nuts_sampler="nutpie",
                quiet=quiet,
                target_accept=CONFIG["target_accept"],
            )
        idata.posterior.attrs.update(
            pair_id=pair_id,
            pair_type=pair["pair_type"],
            gene_a=pair["gene_a"],
            gene_b=pair["gene_b"],
            config=json.dumps(CONFIG),
            runtime_s=time.time() - start,
        )
        results[pair_id] = idata
    except Exception as e:
        print(f"{pair_id}: {e}")
        results[pair_id] = GeneError(e)

save_results(results, args.outfile)
