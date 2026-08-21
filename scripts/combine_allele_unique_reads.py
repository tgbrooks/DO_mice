import polars as pl
import re

temp = []
for file in snakemake.input.allele_unique:
    mouse_id = re.search("(DO|SIM)[0-9]+", file).group()
    temp.append(pl.read_parquet(file).with_columns(mouse_id=pl.lit(mouse_id)))
res = pl.concat(temp)
res.write_parquet(snakemake.output.allele_unique)
