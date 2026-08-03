"""GBRS: multi-way alignment and allele-specific quantification.

This follows the GBRS pipeline (https://github.com/churchill-lab/gbrs):

    bowtie -> emase bam2emase -> [emase get-common-alignments] -> gbrs compress
        -> gbrs quantify (multi-way)
        -> gbrs quantify (diploid, constrained by the mouse's genotype)

The genotype used for the diploid step is by default converted from the
downloaded array genotypes (steps/genotypes.smk). Set `gbrs: run_reconstruct:`
to also run GBRS's own reconstruction from the RNA-seq (`gbrs reconstruct` +
`interpolate` + `plot` + `export`), and `gbrs: genotype_source: reconstructed`
to quantify against that instead.

Reads are aligned one end at a time, as GBRS requires: bowtie reports all
alignments to the 8-way pooled transcriptome, and the pairing is applied
afterwards on the EMASE matrices by `emase get-common-alignments`.
"""

GBRS_QUANT_SUFFIXES = [
    "genes.tpm",
    "genes.expected_read_counts",
    "genes.alignment_counts",
    "isoforms.tpm",
    "isoforms.expected_read_counts",
    "isoforms.alignment_counts",
]


def end_fastq(wildcards) -> str:
    """The merged FASTQ for one end of one mouse."""
    return f"results/{wildcards.tissue}/fastq/{wildcards.mouse}_{wildcards.end}.fastq.gz"


rule bowtie_align:
    """Align one FASTQ end to the pooled 8-way transcriptome, reporting all hits.

    `-a --best --strata -v 3` is the GBRS alignment recipe: every best-stratum
    alignment is kept so that EMASE can apportion reads across founders.
    """
    input:
        fastq = end_fastq,
        index = BOWTIE_INDEX_FILES,
    output:
        bam = temp("results/{tissue}/gbrs/{mouse}.{end}.bam"),
    params:
        index = BOWTIE_INDEX,
        log = "results/{tissue}/gbrs/{mouse}.{end}.bowtie.log",
    threads: int(GBRS["align_threads"])
    resources:
        mem_mb = 16000,
        runtime = '24h',
    container:
        "images/gbrs.sif"
    shell:
        """
        zcat {input.fastq} \
            | bowtie -p {threads} -q -a --best --strata --sam -v 3 {params.index} - \
                2> {params.log} \
            | samtools view -bS - > {output.bam}
        """


rule bam2emase:
    """Convert one end's alignments into an EMASE incidence matrix."""
    input:
        bam = "results/{tissue}/gbrs/{mouse}.{end}.bam",
        info = gbrs_file("transcript_info"),
    output:
        h5 = temp("results/{tissue}/gbrs/{mouse}.{end}.h5"),
    params:
        haplotypes = HAPLOTYPES,
    resources:
        mem_mb = 32000,
        runtime = '12h',
    container:
        "images/gbrs.sif"
    shell:
        """
        emase bam2emase \
            -i {input.bam} \
            -m {input.info} \
            -h {params.haplotypes} \
            -o {output.h5}
        """


rule emase_common_alignments:
    """Keep only reads whose two ends agree, for paired-end samples."""
    input:
        r1 = "results/{tissue}/gbrs/{mouse}.R1.h5",
        r2 = "results/{tissue}/gbrs/{mouse}.R2.h5",
    output:
        h5 = temp("results/{tissue}/gbrs/{mouse}.merged.h5"),
    resources:
        mem_mb = 64000,
        runtime = '12h',
    container:
        "images/gbrs.sif"
    shell:
        """
        emase get-common-alignments \
            -i {input.r1} \
            -i {input.r2} \
            -o {output.h5}
        """


def emase_h5(wildcards) -> str:
    """The EMASE file to quantify: the paired intersection, or the single end."""
    if is_paired(wildcards.tissue, wildcards.mouse):
        return f"results/{wildcards.tissue}/gbrs/{wildcards.mouse}.merged.h5"
    return f"results/{wildcards.tissue}/gbrs/{wildcards.mouse}.SE.h5"


rule gbrs_compress:
    """Collapse reads with identical alignment patterns."""
    input:
        emase_h5,
    output:
        h5 = "results/{tissue}/gbrs/{mouse}.compressed.h5",
    resources:
        mem_mb = 32000,
        runtime = '12h',
    container:
        "images/gbrs.sif"
    shell:
        "gbrs compress -i {input} -o {output.h5}"


rule gbrs_quantify_multiway:
    """Quantify expression across all 8 founder haplotypes.

    Also the input to `gbrs reconstruct`, which infers the genome from these
    per-founder TPMs.
    """
    input:
        h5 = "results/{tissue}/gbrs/{mouse}.compressed.h5",
        gene2transcripts = gbrs_file("gene2transcripts"),
        lengths = gbrs_file("transcript_lengths"),
    output:
        expand(
            "results/{{tissue}}/gbrs/{{mouse}}.multiway.{suffix}",
            suffix=GBRS_QUANT_SUFFIXES,
        ),
    params:
        outbase = "results/{tissue}/gbrs/{mouse}",
        model = GBRS["multiread_model"],
    resources:
        mem_mb = 32000,
        runtime = '24h',
    container:
        "images/gbrs.sif"
    shell:
        """
        gbrs quantify \
            -i {input.h5} \
            -g {input.gene2transcripts} \
            -L {input.lengths} \
            -M {params.model} \
            -a \
            -o {params.outbase}
        """


rule gbrs_reconstruct:
    """Reconstruct the genome from the multi-way expression (GBRS's own HMM).

    Optional: enabled by `gbrs: run_reconstruct: true`. Useful as a check
    against the array genotypes, and required if `genotype_source` is
    "reconstructed".
    """
    input:
        expr = "results/{tissue}/gbrs/{mouse}.multiway.genes.tpm",
        tprob = lambda w: transition_prob_file(w.mouse),
        avecs = gbrs_file("emissions"),
        gene_pos = gbrs_file("gene_pos"),
    output:
        genoprobs = "results/{tissue}/gbrs/{mouse}.genoprobs.npz",
        genotypes = "results/{tissue}/gbrs/{mouse}.genotypes.tsv",
        genotypes_npz = "results/{tissue}/gbrs/{mouse}.genotypes.npz",
    params:
        outbase = "results/{tissue}/gbrs/{mouse}",
        data_dir = GBRS_DIR,
    resources:
        mem_mb = 16000,
        runtime = '8h',
    container:
        "images/gbrs.sif"
    shell:
        """
        export GBRS_DATA={params.data_dir}
        gbrs reconstruct \
            -e {input.expr} \
            -t {input.tprob} \
            -x {input.avecs} \
            -g {input.gene_pos} \
            -o {params.outbase}
        """


rule gbrs_quantify_diploid:
    """Quantify expression against the mouse's own diploid genome.

    The genotype calls come either from the downloaded array genotypes or from
    `gbrs reconstruct`, per `gbrs: genotype_source:` in the config.
    """
    input:
        h5 = "results/{tissue}/gbrs/{mouse}.compressed.h5",
        gene2transcripts = gbrs_file("gene2transcripts"),
        lengths = gbrs_file("transcript_lengths"),
        genotypes = lambda w: genotype_file(w.tissue, w.mouse),
    output:
        expand(
            "results/{{tissue}}/gbrs/{{mouse}}.diploid.{suffix}",
            suffix=GBRS_QUANT_SUFFIXES,
        ),
    params:
        outbase = "results/{tissue}/gbrs/{mouse}",
        model = GBRS["multiread_model"],
    resources:
        mem_mb = 32000,
        runtime = '24h',
    container:
        "images/gbrs.sif"
    shell:
        """
        gbrs quantify \
            -i {input.h5} \
            -g {input.gene2transcripts} \
            -L {input.lengths} \
            -G {input.genotypes} \
            -M {params.model} \
            -a \
            -o {params.outbase}
        """


rule gbrs_interpolate:
    """Put the reconstructed genotype probabilities on the uniform genome grid."""
    input:
        genoprobs = "results/{tissue}/gbrs/{mouse}.genoprobs.npz",
        grid = gbrs_file("genome_grid"),
        gene_pos = gbrs_file("gene_pos"),
    output:
        "results/{tissue}/gbrs/{mouse}.interpolated.genoprobs.npz",
    params:
        data_dir = GBRS_DIR,
    resources:
        mem_mb = 16000,
        runtime = '4h',
    container:
        "images/gbrs.sif"
    shell:
        """
        export GBRS_DATA={params.data_dir}
        gbrs interpolate \
            -i {input.genoprobs} \
            -g {input.grid} \
            -p {input.gene_pos} \
            -o {output}
        """


rule gbrs_plot:
    """Plot the reconstructed founder mosaic for one mouse."""
    input:
        "results/{tissue}/gbrs/{mouse}.interpolated.genoprobs.npz",
    output:
        "results/{tissue}/gbrs/{mouse}.genome.pdf",
    params:
        data_dir = GBRS_DIR,
    resources:
        mem_mb = 8000,
        runtime = '1h',
    container:
        "images/gbrs.sif"
    shell:
        """
        export GBRS_DATA={params.data_dir}
        gbrs plot -i {input} -o {output} -n {wildcards.mouse}
        """


rule gbrs_export:
    """Export founder dosages at the grid positions, for QTL mapping."""
    input:
        genoprobs = "results/{tissue}/gbrs/{mouse}.interpolated.genoprobs.npz",
        grid = gbrs_file("genome_grid"),
    output:
        "results/{tissue}/gbrs/{mouse}.interpolated.genoprobs.tsv",
    params:
        haplotypes = HAPLOTYPES,
        data_dir = GBRS_DIR,
    resources:
        mem_mb = 8000,
        runtime = '1h',
    container:
        "images/gbrs.sif"
    shell:
        """
        export GBRS_DATA={params.data_dir}
        gbrs export \
            -i {input.genoprobs} \
            -s {params.haplotypes} \
            -g {input.grid} \
            -o {output}
        """


rule combine_tpm:
    """Combine the per-mouse gene TPMs of a tissue into one gene x mouse table.

    Total (both haplotypes) TPM per gene, plus a per-founder table for
    allele-specific analyses.
    """
    input:
        tpm = lambda w: expand(
            "results/{tissue}/gbrs/{mouse}.{mode}.genes.tpm",
            tissue=w.tissue,
            mouse=MICE[w.tissue],
            mode=w.mode,
        ),
    output:
        total = "results/{tissue}/{tissue}.{mode}.genes.tpm.parquet",
        by_founder = "results/{tissue}/{tissue}.{mode}.genes.founder_tpm.parquet",
    params:
        mice = lambda w: MICE[w.tissue],
        haplotypes = HAP_LIST,
    resources:
        mem_mb = 8000,
    script:
        "../scripts/combine_tpm.py"

rule combine_counts:
    """Combine the per-mouse gene counts of a tissue into one gene x mouse table.

    Total (both haplotypes) counts per gene, plus a per-founder table for
    allele-specific analyses.
    """
    input:
        counts = lambda w: expand(
            "results/{tissue}/gbrs/{mouse}.{mode}.genes.expected_read_counts",
            tissue=w.tissue,
            mouse=MICE[w.tissue],
            mode=w.mode,
        ),
    output:
        total = "results/{tissue}/{tissue}.{mode}.genes.expected_read_counts.parquet",
        by_founder = "results/{tissue}/{tissue}.{mode}.genes.founder_expected_read_counts.parquet",
    params:
        mice = lambda w: MICE[w.tissue],
        haplotypes = HAP_LIST,
    resources:
        mem_mb = 8000,
    script:
        "../scripts/combine_counts.py"
