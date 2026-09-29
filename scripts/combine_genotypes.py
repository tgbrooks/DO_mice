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
            "gene_id",
            mouse_id=pl.lit(mouse),
            genotype=pl.col("diplotype"),
            genotype_confidence=pl.col("confidence"),
            genotype_n_markers=pl.col("n_markers"),
        )
    )

pl.concat(temp).write_parquet(snakemake.output.geno)
