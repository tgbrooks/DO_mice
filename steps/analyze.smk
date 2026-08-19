"""
Rules relating to our analyses and models
"""

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

rule model_buffering:
    """ Compute buffering factors for each gene of a chromosome """
    input:
        counts = "results/{tissue}/{tissue}.diploid.genes.founder_expected_read_counts.parquet",
        allele_unique = lambda wildcards: expand(
            f"processed/{{tissue}}/gbrs_allele_unique_reads/{mouse_id}.allele_unique_reads.parquet",
            mouse_id = MICE[wildcards.tissue]
        ),
        size_factors = "results/{tissue}/size_factors.txt",
        annot = "processed/gene_annot.txt",
        phenotypes = "phenotypes.csv.gz",
        genotypes = "results/genotypes.parquet",
        kinship = "geno/kinship/{chromosome}.txt",
    output:
        outfile = "processed/{tissue}/buffering/{chromosome}.txt"
    params:
        min_median_counts = config['MIN_MEDIAN_COUNTS'],
    resources:
        mem_mb = 18_000,
    container:
        "images/rgeneral.sif"
    script:
        "../scripts/model_buffering.R"

rule collect_buffering_results:
    input:
        results = expand("processed/{{tissue}}/buffering/{chromosome}.txt", chromosome = [c for c in config['chromosomes'] if c != 'X'])
    output:
        results = "results/{tissue}/buffering.txt"
    resources:
        mem_mb = 6_000
    run:
        import polars as pl
        temp = [pl.read_csv(r, separator="\t") for r in input.results]
        pl.concat(temp).write_csv(output.results, separator="\t")
