"""
Rules relating to our analyses and models
"""
import pathlib

rule gene_annot:
    """ Simplified gene annotation for easy consumption """
    input:
        gtf = config['gtf'],
    output:
        annot = 'processed/gene_annot.txt'
    resources:
        mem_mb = 6_000,
    run:
        import polars as pl
        import polars_bio as pb

        annot = pb.scan_gtf(input.gtf, attr_fields=["gene_id", "gene_name"]) \
            .filter(pl.col("type") == "gene") \
            .collect() \
            .select("gene_id", "gene_name", "chrom", 'start', 'end', 'strand') \
            .write_csv(output.annot, separator="\t")

rule compute_size_factors:
    """ Compute DESeq2 size factors for each library """
    input:
        counts = "results/{tissue}/{tissue}.diploid.genes.founder_expected_read_counts.parquet",
    output:
        outfile = "results/{tissue}/size_factors.txt",
    resources:
        mem_mb = 6_000
    container:
        "images/rgeneral.sif"
    script:
        "../scripts/compute_size_factors.R"

checkpoint chunk_chromosomes:
    """ Break each chromosome into 100-gene chunks """
    input:
        annot = config['gtf'],
    output:
        outdir = directory('processed/{tissue}/chromosome_chunks')
    params:
        chromosomes = config['chromosomes'],
    run:
        import pathlib
        import polars as pl
        import polars_bio as pb
        outdir = pathlib.Path(output.outdir)
        outdir.mkdir(exist_ok=True)
        annot = pb.scan_gtf(input.annot, attr_fields=["gene_id"]).filter(pl.col('type') == 'gene').collect()
        for ((chrom,), chrom_data) in annot.group_by("chrom"):
            if chrom not in params.chromosomes:
                continue
            for i,chunk in enumerate(chrom_data.iter_slices(n_rows=100)):
                chunk.select('gene_id').write_csv(outdir / f"{chrom}.{i}.genes.txt")

def get_chunk_genes(wildcards):
    chunkdir = pathlib.Path(checkpoints.chunk_chromosomes.get(tissue=wildcards.tissue).output.outdir)
    import polars as pl
    genes = pl.read_csv(chunkdir / f"{wildcards.chromosome}.{wildcards.chunk_num}.genes.txt", separator="\t")
    return list(genes['gene_id'])

rule model_buffering:
    """ Compute buffering factors for each gene of a chromosome """
    input:
        counts = "results/{tissue}/{tissue}.diploid.genes.founder_expected_read_counts.parquet",
        allele_unique_reads = "processed/{tissue}/allele_unique_reads.parquet",
        size_factors = "results/{tissue}/size_factors.txt",
        phenotypes = "phenotypes.csv.gz",
        genotypes = "results/genotypes.parquet",
        kinship = "geno/kinship/{chromosome}.txt",
        chunks = lambda wildcards: checkpoints.chunk_chromosomes.get(tissue=wildcards.tissue).output.outdir # indicates we need the checkpoint for params.genes
    output:
        outfile = "processed/{tissue}/buffering/{chromosome}.{chunk_num}.txt"
    params:
        min_median_counts = config['MIN_MEDIAN_COUNTS'],
        genes = get_chunk_genes,
    resources:
        mem_mb = 18_000,
    container:
        "images/rgeneral.sif"
    script:
        "../scripts/model_buffering.R"

def get_all_model_chunk_results(wildcards):
    chunkdir = pathlib.Path(checkpoints.chunk_chromosomes.get(tissue=wildcards.tissue).output.outdir)
    temp = []
    for file in chunkdir.glob("*.genes.txt"):
        chromosome, chunk_num, *_ = file.name.split(".")
        if chromosome == "X":
            continue
        temp.append(f"processed/{wildcards.tissue}/buffering/{chromosome}.{chunk_num}.txt")
    return temp

rule collect_buffering_results:
    input:
        results = get_all_model_chunk_results,
    output:
        results = "results/{tissue}/buffering.txt"
    resources:
        mem_mb = 6_000
    run:
        import polars as pl
        temp = [pl.read_csv(r, separator="\t", null_values="NA") for r in input.results]
        pl.concat(temp).write_csv(output.results, separator="\t")
