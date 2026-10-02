"""
Rules relating to our analyses and models
"""
import pathlib

rule gene_annot:
    """ Simplified gene annotation for easy consumption """
    input:
        gtf = "gbrs_ref/v116/reference.gtf.gz",
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
        counts = "processed/{tissue}/allele_unique_reads.parquet"
    output:
        outfile = "processed/{tissue}/size_factors.txt",
    resources:
        mem_mb = 6_000
    container:
        "images/rgeneral.sif"
    script:
        "../scripts/compute_size_factors.R"

checkpoint chunk_chromosomes:
    """ Break each chromosome into 100-gene chunks """
    input:
        annot = "gbrs_ref/v116/reference.gtf.gz",
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
    chunkdir = pathlib.Path(checkpoints.chunk_chromosomes.get(tissue=wildcards.tissue_real).output.outdir)
    import polars as pl
    genes = pl.read_csv(chunkdir / f"{wildcards.chromosome}.{wildcards.chunk_num}.genes.txt", separator="\t")
    return list(genes['gene_id'])

rule model_buffering:
    """ Compute buffering factors for each gene of a chromosome """
    input:
        #counts = "processed/{tissue_real}/{tissue_real}.diploid.genes.founder_expected_read_counts.parquet",
        allele_unique_reads = "processed/{tissue_real}/allele_unique_reads.parquet",
        size_factors = "processed/{tissue_real}/size_factors.txt",
        phenotypes = "phenotypes.csv.gz",
        genotypes = "processed/genotypes.parquet",
        kinship = "geno/kinship/{chromosome}.txt",
        chunks = lambda wildcards: checkpoints.chunk_chromosomes.get(tissue=wildcards.tissue_real).output.outdir # indicates we need the checkpoint for params.genes
    output:
        outfile = "processed/{tissue_real}/buffering/{chromosome}.{chunk_num}.txt"
    params:
        min_median_counts = config['MIN_MEDIAN_COUNTS'],
        genes = get_chunk_genes,
        outlier_ids = config['outlier_ids'], # Remove these mice
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
        temp =[]
        for r in input.results:
            try:
                data = pl.read_csv(r, separator="\t", null_values="NA", schema_overrides={"binom_p_gof": pl.Float64})
            except pl.exceptions.NoDataError:
                print(f"No results in {r} - skipping")
                continue
            temp.append(data)
        pl.concat(temp).write_csv(output.results, separator="\t")

rule model_buffering_with_bias:
    """ Run bias-correcting ASE models and total count models """
    input:
        allele_unique_reads = "processed/{tissue_real}/allele_unique_reads.parquet",
        size_factors = "processed/{tissue_real}/size_factors.txt",
        phenotypes = "phenotypes.csv.gz",
        chunks = lambda wildcards: checkpoints.chunk_chromosomes.get(tissue=wildcards.tissue_real).output.outdir # indicates we need the checkpoint for params.genes
    output:
        outfile = "processed/{tissue_real}/buffering_with_bias/{chromosome}.{chunk_num}.txt"
    params:
        min_median_counts = config['MIN_MEDIAN_COUNTS'],
        genes = get_chunk_genes,
        outlier_ids = config['outlier_ids'], # Remove these mice
    resources:
        mem_mb = 24_000,
        threads = 4,
    container:
        "images/rgeneral.sif"
    script:
        "../scripts/model_buffering_with_bias.py"


rule compute_coverage:
    input:
        R1 = "processed/{tissue}/gbrs/{sample_id}.R1.bam",
        R2 = "processed/{tissue}/gbrs/{sample_id}.R2.bam",
        emase = "processed/{tissue}/gbrs/{sample_id}.compressed.h5",
    output:
        out = "processed/{tissue}/cov/{sample_id}.cov.parquet",
    resources:
        mem_mb = 24_000,
    shell:
        "python ../scripts/compute_coverage.py --R1 {input.R1} --R2 {input.R2} --emase {input.emase} --out {output}"


def fragment_length_genotype_args(wildcards) -> str:
    """Simulated 'mice' are one founder throughout; real mice use their genotypes."""
    if wildcards.tissue == "simulated_reads":
        return f"--homozygous-for {wildcards.mouse}"
    return f"--genotypes geno/gbrs_genotypes/{wildcards.mouse}.confidence.tsv"

rule fragment_lengths:
    """ Estimate a sample's fragment length distribution from read pairs on
    long, single-isoform transcripts of genes homozygous in that mouse """
    input:
        R1 = "processed/{tissue}/gbrs/{mouse}.R1.bam",
        R2 = "processed/{tissue}/gbrs/{mouse}.R2.bam",
        gtf = "gbrs_ref/v116/reference.gtf.gz",
        genotypes = lambda wildcards: [] if wildcards.tissue == "simulated_reads"
            else f"geno/gbrs_genotypes/{wildcards.mouse}.confidence.tsv",
    output:
        out = "processed/{tissue}/frag_dist/{mouse}.json",
    params:
        genotype_args = fragment_length_genotype_args,
    resources:
        mem_mb = 36_000,
        runtime = '6h',
    shell:
        "python scripts/fragment_lengths.py --R1 {input.R1} --R2 {input.R2} --gtf {input.gtf} {params.genotype_args} --out {output.out}"

rule gather_fragment_lengths:
    """ Fragment length distribution of all samples """
    input:
        frag_dist = lambda w: expand(
            "processed/{{tissue}}/frag_dist/{mouse}.json",
            mouse=MICE[w.tissue],
        ),
    output:
        out = "processed/{tissue}/frag_dist.txt",
    run:
        import json
        import polars as pl
        import pathlib
        temp = []
        for f in input.frag_dist:
            f = pathlib.Path(f)
            mouse_id = f.name.split(".")[0]
            with open(f) as frag_dist:
                data = json.load(frag_dist)
            temp.append({
                "mouse_id": mouse_id,
                "frag_len_mean": data['mean'],
                "frag_len_median": data['median'],
                "frag_len_sd": data['sd'],
                "frag_len_q0.01": data['quantiles']["0.01"],
                "frag_len_q0.99": data['quantiles']["0.99"],
            })
        pl.DataFrame(temp).write_csv(output.out, separator="\t")

rule salmon:
    """ Run salmon on the 8x founders transcriptome reference - to get effective lengths of transcripts """
    input:
        R1 = "processed/{tissue}/fastq/{mouse}_R1.fastq.gz",
        R2 = "processed/{tissue}/fastq/{mouse}_R2.fastq.gz",
        index = "gbrs_ref/v116/salmon_all_haps/",
    output:
        outdir = directory("processed/{tissue}/salmon/{mouse}"),
    threads: 6
    resources:
        mem_mb = 6000,
        runtime = '3h',
    container:
        "docker://combinelab/salmon:2.8.0"
    shell:
        """ salmon quant --index {input.index} -1 {input.R1} -2 {input.R2} --output {output.outdir} -p {threads} --seqBias --gcBias --posBias """

rule gather_effective_lengths:
    input:
        data_dirs = lambda w: expand("processed/{{tissue}}/salmon/{mouse}", mouse=MICE[w.tissue]),
        index = "gbrs_ref/v116/salmon_all_haps/",
        geno = "processed/genotypes.parquet",
        gtf = "gbrs_ref/v116/reference.gtf.gz",
    output:
        "processed/{tissue}/effective_lengths.parquet"
    resources:
        mem_mb=24_000
    script:
        "../scripts/gather_effective_lengths.py"
