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
        counts = "processed/{tissue}/allele_unique_reads.parquet",
        effective_lengths = "processed/{tissue}/effective_lengths.by_gene.parquet",
    output:
        outfile = "processed/{tissue}/size_factors.txt",
        out_normfile = "processed/{tissue}/normalization_factors.txt",
    resources:
        mem_mb = 12_000
    container:
        "images/rgeneral.sif"
    shell:
        "Rscript scripts/compute_size_factors.R {input.counts} {input.effective_lengths} {output.outfile} {output.out_normfile}"

checkpoint chunk_chromosomes:
    """ Break each chromosome into 100-gene chunks """
    input:
        annot = "gbrs_ref/v116/reference.gtf.gz",
    output:
        outdir = directory('processed/{tissue_real}/chromosome_chunks')
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
    chunkdir = pathlib.Path(checkpoints.chunk_chromosomes.get(tissue_real=wildcards.tissue_real).output.outdir)
    import polars as pl
    genes = pl.read_csv(chunkdir / f"{wildcards.chromosome}.{wildcards.chunk_num}.genes.txt", separator="\t")
    return list(genes['gene_id'])

def get_all_model_chunk_results(dir):
    def inner(wildcards):
        chunkdir = pathlib.Path(checkpoints.chunk_chromosomes.get(tissue_real=wildcards.tissue_real).output.outdir)
        temp = []
        for file in chunkdir.glob("*.genes.txt"):
            chromosome, chunk_num, *_ = file.name.split(".")
            if chromosome == "X":
                continue
            temp.append(f"processed/{wildcards.tissue_real}/{dir}/{chromosome}.{chunk_num}.results.parquet")
        return temp
    return inner

def get_chunk_file(wildcards):
    chunkdir = pathlib.Path(checkpoints.chunk_chromosomes.get(tissue_real=wildcards.tissue_real).output.outdir)
    return chunkdir / f"{wildcards.chromosome}.{wildcards.chunk_num}.genes.txt"

rule run_ase_model_part1:
    """ Run ASE part 1 model to compute sample-level allelic-imbalance """
    input:
        allele_unique_reads = "processed/{tissue_real}/allele_unique_reads.parquet",
        effective_lengths = "processed/{tissue_real}/effective_lengths.by_gene.parquet",
        gene_list = get_chunk_file
    output:
        output = "processed/{tissue_real}/buffering/{chromosome}.{chunk_num}.ase_model_part1.nc"
    threads: 4
    resources:
        mem_mb=24_000
    shell:
        "python scripts/run_ase_model_part1.py --tissue {wildcards.tissue_real} --outfile {output} --gene_ids_file {input.gene_list}"

rule run_ase_model_part2:
    """ Compute haplotype-level beta effects and summarize"""
    input:
        allele_unique_reads = "processed/{tissue_real}/allele_unique_reads.parquet",
        effective_lengths = "processed/{tissue_real}/effective_lengths.by_gene.parquet",
        part1 = "processed/{tissue_real}/buffering/{chromosome}.{chunk_num}.ase_model_part1.nc",
        gene_list = get_chunk_file
    output:
        "processed/{tissue_real}/buffering/{chromosome}.{chunk_num}.ase_model_part2.json"
    resources:
        mem_mb=24_000
    shell:
        "python scripts/run_ase_model_part2.py --tissue {wildcards.tissue_real} --outfile {output} --gene_ids_file {input.gene_list} --part1 {input.part1}"

rule run_total_model:
    """ Compute model of total counts """
    input:
        allele_unique_reads = "processed/{tissue_real}/allele_unique_reads.parquet",
        size_factors = "processed/{tissue_real}/size_factors.txt",
        phenotypes = "phenotypes.csv.gz",
        effective_lengths = "processed/{tissue_real}/effective_lengths.by_gene.parquet",
        gene_list = get_chunk_file,
    output:
        "processed/{tissue_real}/buffering/{chromosome}.{chunk_num}.total_model.nc",
    threads: 4
    resources:
        mem_mb=24_000
    shell:
        "python scripts/run_total_model.py --tissue {wildcards.tissue_real} --outfile {output} --gene_ids_file {input.gene_list}"


rule run_buffering_model:
    """ Combine ASE and total models to model buffering"""
    input:
        ase_part1 = "processed/{tissue_real}/buffering/{chromosome}.{chunk_num}.ase_model_part1.nc",
        ase_part2 = "processed/{tissue_real}/buffering/{chromosome}.{chunk_num}.ase_model_part2.json",
        total = "processed/{tissue_real}/buffering/{chromosome}.{chunk_num}.total_model.nc",
        gene_list = get_chunk_file,
    output:
        outfile = "processed/{tissue_real}/buffering/{chromosome}.{chunk_num}.results.parquet"
    resources:
        mem_mb = 16_000,
    shell:
        "python scripts/run_buffering_model.py --tissue {wildcards.tissue_real} --outfile {output} --gene_ids_file {input.gene_list} --ase_part1 {input.ase_part1} --ase_part2 {input.ase_part2} --total {input.total}"

rule collect_buffering_results:
    input:
        results = get_all_model_chunk_results("buffering"),
    output:
        results = "results/{tissue_real}/buffering_full_results.parquet"
    resources:
        mem_mb = 18_000
    run:
        import polars as pl
        data = pl.DataFrame([x for infile in input.results for x in pl.read_parquet(infile).to_dicts()])
        data.write_parquet(output.results)


rule compute_coverage:
    input:
        R1 = "processed/{tissue}/gbrs/{sample_id}.R1.bam",
        R2 = "processed/{tissue}/gbrs/{sample_id}.R2.bam",
        emase = "processed/{tissue}/gbrs/{sample_id}.compressed.h5",
    output:
        out = "processed/{tissue}/cov/{sample_id}.cov.parquet",
    threads: 8
    resources:
        mem_mb = 48_000,
    shell:
        "python scripts/compute_coverage.py --R1 {input.R1} --R2 {input.R2} --emase {input.emase} --out {output}"

rule gather_fragment_lengths:
    """ Fragment length distribution of all samples from Salmon """
    input:
        salmon_dir = lambda w: expand(
            "processed/{{tissue}}/salmon/{mouse}/",
            mouse=MICE[w.tissue],
        ),
    output:
        out = "processed/{tissue}/frag_dist.txt",
    run:
        import polars as pl
        import numpy as np
        import gzip
        import pathlib
        temp = []
        for sdir in input.salmon_dir:
            sdir = pathlib.Path(sdir)
            mouse_id = sdir.name
            flen = pl.read_csv(sdir / "libParams/flenDist.txt", separator="\t", has_header=False).to_numpy()[0]
            pos = np.arange(len(flen))
            mean = np.sum(pos * flen) / flen.sum()
            cc = np.cumsum(flen)
            median = np.searchsorted(cc, flen.sum()/2)
            var = np.sum((pos - mean)**2 * flen) / flen.sum()
            temp.append({
                "mouse_id": mouse_id,
                "frag_len_mean": float(mean),
                "frag_len_median": int(median),
                "frag_len_sd": float(np.sqrt(var)),
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
        data_dirs = lambda w: expand("processed/{{tissue_real}}/salmon/{mouse}", mouse=MICE[w.tissue_real]),
        index = "gbrs_ref/v116/salmon_all_haps/",
        geno = "processed/genotypes.parquet",
        gtf = "gbrs_ref/v116/reference.gtf.gz",
    output:
        "processed/{tissue_real}/effective_lengths.parquet",
        "processed/{tissue_real}/effective_lengths.by_gene.parquet",
    resources:
        mem_mb=24_000
    script:
        "../scripts/gather_effective_lengths.py"
