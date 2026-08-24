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
