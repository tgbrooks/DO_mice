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
