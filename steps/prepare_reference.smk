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

#gffread -w ${S}.transcripts.fa -g ${S}_v3.fa ${S}.gff3_polished
