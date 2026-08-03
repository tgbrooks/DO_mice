"""Combine per-mouse GBRS gene counts files into tissue-level tables.

Run as a Snakemake `script:` (see the `combine_tpm` rule), so it reads its
inputs from the `snakemake` object. GBRS writes one file per mouse with a
`locus` column, one column per founder haplotype, a `total` column, and (in
diploid mode) a `notes` column holding the called diplotype.

Two tables are written:

- `{tissue}.{mode}.genes.tpm.parquet`: total counts, one column per mouse.
- `{tissue}.{mode}.genes.founder_tpm.parquet`: long form with the per-founder counts
  of every gene in every mouse, for allele-specific analyses.
"""

import polars as pl

paths = list(snakemake.input.counts)
mice = list(snakemake.params.mice)
haplotypes = list(snakemake.params.haplotypes)

totals: list[pl.DataFrame] = []
by_founder: list[pl.DataFrame] = []

for mouse, path in zip(mice, paths):
    df = pl.read_csv(path, separator="\t")
    totals.append(
        df.select(pl.col("locus").alias("gene_id"), pl.col("total").alias(mouse))
    )
    by_founder.append(
        df.select(
            pl.col("locus").alias("gene_id"),
            pl.lit(mouse).alias("mouse_id"),
            *[pl.col(h) for h in haplotypes],
            pl.col("total"),
        )
    )

total_table = totals[0]
for df in totals[1:]:
    total_table = total_table.join(df, on="gene_id", how="full", coalesce=True)
total_table = total_table.sort("gene_id")
total_table.write_parquet(snakemake.output.total)

pl.concat(by_founder).sort(["gene_id", "mouse_id"]).write_parquet(
    snakemake.output.by_founder
)
