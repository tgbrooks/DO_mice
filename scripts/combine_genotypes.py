"""
Combine genotypes (diplotypes) of each mouse / gene together into one file
"""

import polars as pl

paths = list(snakemake.input.geno)
mice = list(snakemake.params.mice)

temp: list[pl.DataFrame] = []

for mouse, path in zip(mice, paths):
    df = pl.read_csv(path, separator="\t")
    temp.append(
        df.select(
            pl.col("#Gene_ID").alias("gene_id"),
            mouse_id=pl.lit(mouse),
            genotype=pl.col("Diplotype"),
        )
    )

pl.concat(temp).write_parquet(snakemake.output.geno)
