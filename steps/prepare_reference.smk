"""
Ensembl provides reference files for all 8 of our founder strains.
However, they each have their own transcriptome with separate gene IDs,
transcripts, etc. Differences in exon structure would make them incomparable
for ASE. Instead, we map the GRCm39 reference over to the other 7 directly,
giving a more comparable transcriptome.
"""

rule make_chroms_file:
    # Tells liftoff to work chromosome-by-chromosome
    output:
        chroms_file = "gbrs_ref/v116/chroms.txt",
    run:
        import polars as pl
        chroms = config['chromosomes']
        pl.DataFrame({"ref": chroms, "hap": chroms}).write_csv(output.chroms_file, include_header=False)

rule gunzip_fa:
    input:
        "{file}.fa.gz"
    output:
        "{file}.fa"
    shell:
        "zcat {input} > {output}"

rule liftoff:
    input:
        hap_dna = "gbrs_ref/v116/{nonref_haplotype}.dna.fa",
        # B is the reference genome C57Bl/6J
        ref_dna = "gbrs_ref/v116/B.dna.fa",
        ref_gff = "gbrs_ref/v116/B.gff3.gz",
        chroms_file = "gbrs_ref/v116/chroms.txt",
    output:
        hap_gff = "gbrs_ref/v116/{nonref_haplotype}.gff3",
        hap_gff_polished = "gbrs_ref/v116/{nonref_haplotype}.gff3_polished",
        unmapped = "gbrs_ref/v116/{nonref_haplotype}.unmapped.txt",
    resources:
        threads=16,
        mem_mb=64_000,
    container:
        "images/liftoff.sif"
    shell:
        """
        liftoff -g {input.ref_gff} \
                -u {output.unmapped} \
                -o {output.hap_gff} \
                -chroms {input.chroms_file} \
                -exclude_partial \
                -p 16 -polish \
                -dir gbrs_ref/v116/temp.{wildcards.nonref_haplotype}/ \
                {input.hap_dna} \
                {input.ref_dna}
        """

rule gzip_gff:
    input:
        "{file}.gff3"
    output:
        "{file}.gff3.gz"
    shell:
        "gzip -c {input} > {output}"

rule select_transcripts:
    input:
        gffs = expand("gbrs_ref/v116/{haplotype}.gff3.gz", haplotype=HAP_LIST)
    output:
        transcripts = "gbrs_ref/v116/selected_transcripts.txt"
    resources:
        mem_mb=36_000
    script:
        "../scripts/select_transcripts.py"

rule extract_transcriptome_file:
    input:
        gff = "gbrs_ref/v116/{haplotype}.gff3.gz",
        fasta = "gbrs_ref/v116/{haplotype}.dna.fa.gz",
        tx = "gbrs_ref/v116/selected_transcripts.txt",
    output:
        fasta = "gbrs_ref/v116/{haplotype}.cdna.fa.gz"
    resources:
        mem_mb = 18_000,
    script:
        "../scripts/extract_transcriptome.py"

rule combined_transcriptome:
    input:
        fastas = expand("gbrs_ref/v116/{haplotype}.cdna.fa.gz", haplotype=HAP_LIST),
    output:
        out = "gbrs_ref/v116/all_haps.cdna.fa"
    run:
        import polars as pl
        import polars_bio as pb
        temp = []
        for fasta, hap in zip(input.fastas, HAP_LIST):
            temp.append(pb.read_fasta(fasta).with_columns(
                name = pl.col("name") + "_" + hap
            ))
        pb.write_fasta(pl.concat(temp), output.out)

rule make_bowtie_index:
    input:
        fasta = "gbrs_ref/v116/all_haps.cdna.fa"
    output:
        outfiles = multiext("gbrs_ref/v116/bowtie_index/all_haps", ".1.ebwt", ".2.ebwt", ".3.ebwt", ".4.ebwt", ".rev.1.ebwt", ".rev.2.ebwt")
    container:
        "images/gbrs.sif"
    resources:
        mem_mb = 64_000,
        threads = 12,
    shell:
        "bowtie-build {input.fasta} gbrs_ref/v116/bowtie_index/all_haps -f --threads {threads} --seed 100"

rule make_transcript_info_file:
    input:
        gff="gbrs_ref/v116/B.gff3.gz"
    output:
        out="gbrs_ref/v116/emase.fullTranscripts.info",
    resources:
        mem_mb=12_000
    script:
        "../scripts/make_transcript_info_file.py"

rule make_reference_gtf:
    input:
        gtf = "gbrs_ref/v116/B.gtf.gz"
    output:
        gtf = "gbrs_ref/v116/reference.gtf.gz"
    resources:
        mem_mb=12_000
    script:
        "../scripts/make_reference_gtf.py"
