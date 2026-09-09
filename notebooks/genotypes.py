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
    import numpy as np
    import pathlib
    import math

    return lp, math, mo, np, pathlib, pb, pl, yaml


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
    annot = pb.scan_gtf("gbrs_ref/v116/reference.gtf.gz", attr_fields=["gene_id", "gene_name"]).filter(pl.col("type") == "gene").collect()
    return annot, config


@app.cell
def _(pb, pl):
    tx_annot = pb.scan_gtf("gbrs_ref/v116/reference.gtf.gz", attr_fields=["gene_id", "transcript_id"]).filter(pl.col("type") == "transcript").collect()
    return (tx_annot,)


@app.cell
def _(config, lp):
    HAPLOTYPES = config["haplotypes"].split(",")
    HAPLOTYPE_COLORS = lp.scale_color_brewer(palette="Dark2").palette( len(HAPLOTYPES) )
    OUTLIER_MOUSE_IDS = config['outlier_ids']
    return HAPLOTYPES, HAPLOTYPE_COLORS


@app.cell
def _(HAPLOTYPES, annot, mo):
    gene_ids = annot.sort("gene_id")['gene_id']
    gene_names = annot.sort("gene_id")['gene_name']
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
    HAPLOTYPES,
    HAPLOTYPE_COLORS,
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
    _plt = (
        lp.ggplot(nearby, lp.aes("pos", "fraction", color="haplotype", fill="haplotype"))
        + lp.geom_bar(stat="identity")
        + lp.facet_wrap("mouse_id", ncol=2)
        + lp.ggsize(900, math.ceil(len(_selected_mice)/2)*150)
        + lp.geom_vline(xintercept=gene_start, color="black")
        + lp.geom_vline(xintercept=gene_end, color="black")
        + lp.scale_color_manual(breaks=HAPLOTYPES, values=HAPLOTYPE_COLORS)
        + lp.scale_fill_manual(breaks=HAPLOTYPES, values=HAPLOTYPE_COLORS)
    )
    mo.vstack([
        "Plot of genotypes at markers near the selected gene in mice of a selected diplotype. Black lines denote the gene start/end locations.",
        _plt,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sequence differences
    """)
    return


@app.cell
def _(pb, pl):
    # Assess the differences in transcriptomes
    transcripts = (
        pb.scan_fasta("gbrs_ref/v116/all_haps.cdna.fa")
        .select(
            transcript_id=pl.col("name").str.split("_").list.get(0),
            haplotype=pl.col("name").str.split("_").list.get(1),
            sequence = "sequence",
        )
        .collect()
    )
    return (transcripts,)


@app.cell
def _(gene_selector, lp, plot_msa, transcripts, tx_annot):
    import pyabpoa as pa
    gene_transcripts = tx_annot.filter(gene_id = gene_selector.value)['transcript_id']
    _temp = []
    for transcript_id in gene_transcripts:
        _dat = transcripts.filter(transcript_id = transcript_id)
        a = pa.msa_aligner()
        seqs=list(_dat['sequence'])
        a_res=a.msa(seqs, out_cons=True, out_msa=True) # perform multiple sequence alignment 
        _temp.append(plot_msa(a_res) + lp.ggtitle(transcript_id))
    lp.gggrid(_temp, ncol=1) + lp.ggsize(900, 250*len(_temp)) + lp.ggtb()
    return gene_transcripts, pa


@app.cell
def _(HAPLOTYPES, gene_transcripts, lp, np, pa, pl, transcripts):
    def _():
        num_unique = []
        for transcript_id in gene_transcripts:
            _dat = transcripts.filter(transcript_id = transcript_id)
            a = pa.msa_aligner()
            seqs=list(_dat['sequence'])
            a_res=a.msa(seqs, out_cons=True, out_msa=True) # perform multiple sequence alignment
            for i, hap1 in enumerate(HAPLOTYPES):
                for j, hap2 in enumerate(HAPLOTYPES):
                    seq1 = np.array(list(a_res.msa_seq[i]))
                    seq2 = np.array(list(a_res.msa_seq[j]))
                    seq1_unique_bases = (seq1 != seq2) & (seq1 != "-")
                    seq2_unique_bases = (seq1 != seq2) & (seq2 != "-")

                    num_unique.append({
                        "hap1": hap1,
                        "hap2": hap2,
                        "transcript_id": transcript_id,
                        "num_unique": np.sum(seq1 != seq2),
                        "seq1_unique": np.sum(seq1_unique_bases),
                        "seq2_unique": np.sum(seq2_unique_bases),
                    })
        num_unique = pl.DataFrame(num_unique)
        return (
            lp.ggplot(num_unique, lp.aes(x="hap1", y="hap2", fill="seq1_unique"))
                + lp.facet_grid(y="transcript_id")
                + lp.scale_fill_viridis(option="inferno")
                + lp.scale_y_discrete_reversed()
                + lp.geom_tile(tooltips=lp.layer_tooltips(["hap1", "hap2", "num_unique", "seq1_unique", "seq2_unique"]))
                + lp.ggtitle("Number of unique nucleotide positions")
        )
    _()
    return


@app.cell
def _(HAPLOTYPES, lp, np, pl):
    def plot_msa(a_res):
        assert len(a_res.cons_seq) == 1 # only one consensus sequence, hopefully, not sure when this fails
        cons_seq = np.array(list(a_res.msa_seq[-1]))
        x = np.arange(len(cons_seq))
        hap_num = {hap: 8-i for i, hap in enumerate(HAPLOTYPES)}
        _temp = []
        for hap, seq in zip(HAPLOTYPES, a_res.msa_seq):
            match_cons = np.array(list(seq))  == cons_seq
            is_gap = np.array(list(seq)) == "-"
            _temp.append(pl.DataFrame({
                "pos": x,
                "match_cons": match_cons,
                "is_gap": is_gap,
                "haplotype": hap,
            }))
        msa = pl.concat(_temp) \
            .with_columns(
                type = pl.when(pl.col("is_gap"))
                    .then(pl.lit("gap"))
                    .when(pl.col("match_cons"))
                    .then(pl.lit('match'))
                    .otherwise(pl.lit('mismatch'))
            ).with_columns(
                rle_id = pl.col("type").rle_id().over("haplotype")
            ).group_by(
                ["rle_id", "type", "haplotype"],
            ).agg(
                start = pl.col("pos").min(),
                end = pl.col("pos").max()+1,
            ).with_columns(
                hap_bottom = pl.col("haplotype").replace_strict(hap_num),
                hap_top = pl.col("haplotype").replace_strict(hap_num) + 0.8,
            )
        return (
            lp.ggplot(msa, lp.aes(xmin = "start", xmax="end", ymin = "hap_bottom", ymax="hap_top",fill="type", color="type"))
            + lp.geom_rect()
            + lp.scale_y_continuous(breaks = [x+0.45 for x in hap_num.values()], labels=list(hap_num.keys()))
            + lp.scale_color_manual(breaks=["match", "mismatch", "gap"], values=["black", "red", "white"])
            + lp.scale_fill_manual(breaks=["match", "mismatch", "gap"], values=["black", "red", "white"])
            + lp.ggsize(900, 500)
            + lp.labs(x="pos")
        )

    return (plot_msa,)


if __name__ == "__main__":
    app.run()
