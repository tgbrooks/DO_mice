rule extract_transcriptome:
    """ Extract the transcript sequences from the bowtie index """
    input:
        BOWTIE_INDEX_FILES,
    output:
        "gbrs_ref/transcripts.fasta"
    container:
        "images/gbrs.sif"
    shell:
        "bowtie-inspect gbrs_ref/bowtie.transcripts > {output}"

rule simulate_reads:
    """ Create reads covering each transcript """
    input:
        "gbrs_ref/transcripts.fasta",
    output:
        R1 = "results/simulated_reads/fastq/{haplotype}_R1.fastq.gz",
        R2 = "results/simulated_reads/fastq/{haplotype}_R2.fastq.gz",
    resources:
        mem_mb = 6_000,
    script:
        "../scripts/simulate_reads.py"

rule get_allele_unique_reads_simulated:
    """ Summarize uniques and totals of genes for simulated reads under all genotypes """
    input:
        h5 = expand("results/simulated_reads/gbrs/{haplotype}.compressed.h5", haplotype=HAP_LIST),
    output:
        gene_unique = "processed/simulated_reads/allele_unique_reads.parquet"
    resources:
        mem_mb = 12_000
    script:
        "../scripts/get_allele_unique_reads_simulated.py"

rule count_simulated_reads:
    """ Summarize the generated reads by their true haplotype + source gene """
    input:
        fastq = "results/simulated_reads/fastq/{haplotype}_R1.fastq.gz",
    output:
        by_tx = "results/simulated_reads/source_counts/{haplotype}_by_transcript.txt",
        by_gene = "results/simulated_reads/source_counts/{haplotype}_by_gene.txt",
    resources:
        mem_mb = 12_000,
    script:
        "../scripts/count_simulated_reads.py"

rule combine_simulated_read_counts:
    input:
        by_gene = expand("results/simulated_reads/source_counts/{haplotype}_by_gene.txt", haplotype=HAP_LIST),
        by_tx = expand("results/simulated_reads/source_counts/{haplotype}_by_transcript.txt", haplotype=HAP_LIST),
    output:
        by_gene = "results/simulated_reads/source_counts_by_gene.txt",
        by_tx = "results/simulated_reads/source_counts_by_transcript.txt",
    run:
        import polars as pl
        temp = []
        for file in input.by_gene:
            temp.append(pl.read_csv(file, separator="\t"))
        pl.concat(temp).write_csv(output.by_gene, separator="\t")
        temp = []
        for file in input.by_tx:
            temp.append(pl.read_csv(file, separator="\t"))
        pl.concat(temp).write_csv(output.by_tx, separator="\t")
