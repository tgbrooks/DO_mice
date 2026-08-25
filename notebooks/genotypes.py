import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars_bio as pb
    import yaml
    import lets_plot as lp
    import pathlib
    import math

    return lp, math, mo, pathlib, pb, pl, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inspect genotyping near a selected gene
    """)
    return


@app.cell
def _(pathlib, pl):
    # Load genotypes
    def _():
        temp =[]
        for file in pathlib.Path("geno/alleleprobs/").glob("*.tsv.gz"):
            mouse_id = file.name.split(".")[0]
            temp.append(pl.read_csv(file, separator="\t", schema_overrides={"chr": str}).with_columns(mouse_id = pl.lit(mouse_id)))
        return pl.concat(temp).with_columns(pos = pl.col("marker").str.split("_").list.get(1).cast(int))
    geno = _()
    return (geno,)


@app.cell
def _(pathlib, pl):
    # Load gene genotypes
    def _():
        temp = []
        for file in pathlib.Path("geno/gbrs_genotypes").glob("*.tsv"):
            mouse_id = file.name.split(".")[0]
            temp.append(pl.read_csv(file, separator="\t").with_columns(mouse_id=pl.lit(mouse_id)))
        return pl.concat(temp)
    gene_geno = _().select(gene_id = "#Gene_ID", diplotype="Diplotype", mouse_id="mouse_id")
    return (gene_geno,)


@app.cell
def _(pb, pl, yaml):
    config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
    annot = pb.scan_gtf(config['gtf'], attr_fields=["gene_id", "gene_name"]).filter(pl.col("type") == "gene").collect()
    return (annot,)


@app.cell
def _():
    HAPLOTYPES = list("ABCDEFGH")
    return (HAPLOTYPES,)


@app.cell
def _(HAPLOTYPES, annot, mo):
    gene_ids = annot['gene_id']
    gene_names = annot['gene_name']
    _map = {f"{id} | {name}":id for id,name in zip(gene_ids, gene_names)}
    gene_selector = mo.ui.dropdown(_map, searchable=True)
    _haplotypes = [f"{h1}{h2}" for h1 in HAPLOTYPES for h2 in HAPLOTYPES]
    haplotype_selector = mo.ui.dropdown(_haplotypes, searchable=True)
    mo.hstack([
        gene_selector,
        haplotype_selector,
    ], justify="start")
    return gene_selector, haplotype_selector


@app.cell
def _(annot, gene_selector):
    gene_chrom, gene_start, gene_end = next(annot.filter(gene_id = gene_selector.value).select("chrom", "start", "end").iter_rows())
    return gene_chrom, gene_end, gene_start


@app.cell
def _(
    gene_chrom,
    gene_end,
    gene_geno,
    gene_selector,
    gene_start,
    geno,
    haplotype_selector,
    lp,
    math,
    mo,
    pl,
):
    PADDING = 1_000_000
    _selected_mice = set(gene_geno.filter(
        (pl.col("diplotype") == haplotype_selector.value) | (pl.col("diplotype") == haplotype_selector.value[::-1]),
        gene_id = gene_selector.value,
    )['mouse_id'])
    nearby = geno.filter(
        pl.col("mouse_id").is_in(_selected_mice),
        pl.col("pos") <= gene_end + PADDING,
        pl.col("pos") >= gene_start - PADDING,
        chr = gene_chrom,
    ).unpivot(list("ABCDEFGH"), index=["mouse_id", "pos"], value_name="fraction", variable_name="haplotype")
    _plt = lp.ggplot(nearby, lp.aes("pos", "fraction", color="haplotype", fill="haplotype")) + lp.geom_bar(stat="identity") + lp.facet_wrap("mouse_id", ncol=2) + lp.ggsize(900, math.ceil(len(_selected_mice)/2)*150) + lp.geom_vline(xintercept=gene_start, color="black") + lp.geom_vline(xintercept=gene_end, color="black")
    mo.vstack([
        "Plot of genotypes at markers near the selected gene in mice of a selected diplotype. Black lines denote the gene start/end locations.",
        _plt,
    ])
    return


if __name__ == "__main__":
    app.run()
