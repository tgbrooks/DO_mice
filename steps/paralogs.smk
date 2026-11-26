"""
Paralog compensation: does a genetic effect on gene A's expression move its paralog B
in the opposite direction? For ordered paralog pairs (A, B) on different autosomes, a
model of B's unique reads with A's locus genotype as a predictor (scripts/
run_paralog_model.py), compared with A's own genetic effects from the total-counts model
of the buffering analysis. Matched random gene pairs ("null" pairs) are analysed the
same way. See util/paralogs.py and ase_buffering/paralog_model.py.

Run with:  snakemake paralogs
"""
import pathlib

PARALOGS = config["paralogs"]


rule download_compara_paralogs:
    """Mouse-mouse paralog rows of the Ensembl Compara homologies (same release as the reference)"""
    output:
        "gbrs_ref/v116/compara/mouse_paralogs.tsv.gz"
    params:
        url = PARALOGS["compara_url"],
    resources:
        runtime = '4h',
    shell:
        """
        wget -q -O - {params.url} | zcat \
            | awk -F'\\t' 'NR == 1 {{ for (i = 1; i <= NF; i++) c[$i] = i; print; next }}
                $c["species"] == "mus_musculus" && $c["homology_species"] == "mus_musculus" \
                && ($c["homology_type"] == "within_species_paralog" || $c["homology_type"] == "other_paralog")' \
            | gzip > {output}
        """


checkpoint paralog_pairs:
    """Ordered paralog pairs and matched null pairs for a tissue, split into chunks"""
    input:
        paralogs = "gbrs_ref/v116/compara/mouse_paralogs.tsv.gz",
        gene_annot = "processed/gene_annot.txt",
        class_counts = "processed/{tissue_real}/class_counts.parquet",
    output:
        directory("processed/{tissue_real}/paralogs/pairs")
    resources:
        mem_mb = 24_000,
    shell:
        "python scripts/make_paralog_pairs.py --paralogs {input.paralogs} --gene_annot {input.gene_annot}"
        " --class_counts {input.class_counts} --outdir {output}"


def paralog_pairs_dir(wildcards) -> pathlib.Path:
    return pathlib.Path(checkpoints.paralog_pairs.get(tissue_real=wildcards.tissue_real).output[0])


def paralog_chunk_file(wildcards) -> str:
    return str(paralog_pairs_dir(wildcards) / "chunks" / f"{wildcards.chunk}.tsv")


def paralog_chunk_summaries(wildcards) -> list[str]:
    chunks = sorted((paralog_pairs_dir(wildcards) / "chunks").glob("*.tsv"), key=lambda p: int(p.stem))
    return [f"processed/{wildcards.tissue_real}/paralogs/fits/{p.stem}.json" for p in chunks]


rule run_paralog_model:
    """Fit the paralog model to each pair of a chunk; saves the samples"""
    input:
        chunk = paralog_chunk_file,
        class_counts = "processed/{tissue_real}/class_counts.parquet",
        size_factors = "processed/{tissue_real}/size_factors.txt",
        genotypes = "processed/genotypes.parquet",
        phenotypes = "phenotypes.csv.gz",
    output:
        "processed/{tissue_real}/paralogs/fits/{chunk}.nc"
    wildcard_constraints:
        chunk = r"[0-9]+",
    threads: 4
    resources:
        mem_mb = 16_000,
        runtime = '8h',
    shell:
        "python scripts/run_paralog_model.py --tissue {wildcards.tissue_real} --chunk {input.chunk}"
        " --class_counts {input.class_counts} --outfile {output}"


rule summarize_paralog_model:
    """theta, A's own effects (total model) and the compensation slope per pair"""
    input:
        samples = "processed/{tissue_real}/paralogs/fits/{chunk}.nc",
        chunk = paralog_chunk_file,
        buffering = "results/{tissue_real}/buffering_full_results.parquet",
    output:
        "processed/{tissue_real}/paralogs/fits/{chunk}.json"
    wildcard_constraints:
        chunk = r"[0-9]+",
    resources:
        mem_mb = 16_000,
    shell:
        "python scripts/summarize_paralog_model.py --samples {input.samples} --chunk {input.chunk}"
        " --buffering {input.buffering} --outfile {output}"


rule collect_paralog_results:
    input:
        summaries = paralog_chunk_summaries,
        pairs = lambda w: str(paralog_pairs_dir(w) / "pairs.parquet"),
    output:
        "results/{tissue_real}/paralog_results.parquet"
    resources:
        mem_mb = 16_000,
    run:
        import json
        import polars as pl
        rows = [r for f in input.summaries for r in json.load(open(f))]
        meta = pl.read_parquet(input.pairs).drop("pair_type", "gene_a", "gene_b")
        (
            pl.DataFrame(rows, infer_schema_length=None)
            .join(meta, "pair_id", how="left")
            .write_parquet(output[0])
        )


rule paralogs:
    input:
        expand("results/{tissue}/paralog_results.parquet", tissue=TISSUES),
