"""Downloading raw data: RNA-seq FASTQs from the SRA and the GBRS reference bundle.

Prefetching is kept separate from `fasterq-dump` on purpose: the download is
rate-limited by the custom `sra_downloads` resource (a small pool, set in the
Snakemake profile), while the much slower extraction step is not limited and can
run on as many cores as the scheduler gives it. The prefetched `.sra` data is
`temp()`, so it is deleted as soon as its FASTQs exist.
"""


rule sra_prefetch:
    """Download one SRA run. Consumes a slot of the `sra_downloads` pool."""
    output:
        temp(directory("sra/{run}")),
    resources:
        mem_mb = 4000,
        runtime = '8h',
        # Custom resource limiting concurrent SRA downloads. Set the pool size
        # in the Snakemake profile (`resources: sra_downloads: 3`) or with
        # `--resources sra_downloads=N`.
        sra_downloads = 1,
    container:
        "images/sratools.sif"
    shell:
        """
        rm -rf {output}
        mkdir -p {output}
        prefetch {wildcards.run} --output-directory {output} --max-size u
        """


rule fasterq_dump_paired:
    """Extract a prefetched paired-end run into gzipped R1/R2 FASTQs."""
    input:
        sra = "sra/{run}",
    output:
        r1 = "fastq/{run}_1.fastq.gz",
        r2 = "fastq/{run}_2.fastq.gz",
    threads: 6
    resources:
        mem_mb = 8000,
        runtime = '8h',
    container:
        "images/sratools.sif"
    shell:
        """
        fasterq-dump {input.sra}/{wildcards.run} \
            --split-3 \
            --threads {threads} \
            --outdir {input.sra}
        gzip -c {input.sra}/{wildcards.run}_1.fastq > {output.r1}
        gzip -c {input.sra}/{wildcards.run}_2.fastq > {output.r2}
        """


rule fasterq_dump_single:
    """Extract a prefetched single-end run into one gzipped FASTQ."""
    input:
        sra = "sra/{run}",
    output:
        "fastq/{run}.fastq.gz",
    threads: 6
    resources:
        mem_mb = 8000,
        runtime = '8h',
    container:
        "images/sratools.sif"
    shell:
        """
        fasterq-dump {input.sra}/{wildcards.run} \
            --split-3 \
            --threads {threads} \
            --outdir {input.sra}
        gzip -c {input.sra}/{wildcards.run}.fastq > {output}
        """


def mouse_run_fastqs(wildcards, end: str) -> list[str]:
    """Per-run FASTQ files that make up one mouse's reads for a given end."""
    runs = MOUSE_RUNS[(wildcards.tissue, wildcards.mouse)]
    suffix = {"R1": "_1", "R2": "_2", "SE": ""}[end]
    return [f"fastq/{run}{suffix}.fastq.gz" for run in runs]


rule merge_fastq_paired:
    """Concatenate all of a mouse's paired-end runs into one R1/R2 pair.

    A mouse can have more than one sequencing run; GBRS quantifies one sample at
    a time, so the runs are pooled here.
    """
    input:
        r1 = lambda w: mouse_run_fastqs(w, "R1"),
        r2 = lambda w: mouse_run_fastqs(w, "R2"),
    output:
        r1 = temp("results/{tissue}/fastq/{mouse}_R1.fastq.gz"),
        r2 = temp("results/{tissue}/fastq/{mouse}_R2.fastq.gz"),
    resources:
        runtime = '4h',
    shell:
        """
        cat {input.r1} > {output.r1}
        cat {input.r2} > {output.r2}
        """


rule merge_fastq_single:
    """Concatenate all of a mouse's single-end runs into one FASTQ."""
    input:
        lambda w: mouse_run_fastqs(w, "SE"),
    output:
        temp("results/{tissue}/fastq/{mouse}_SE.fastq.gz"),
    resources:
        runtime = '4h',
    shell:
        "cat {input} > {output}"


rule download_gbrs_reference:
    """Unpack the GBRS supporting-files bundle into the configured data dir.

    Set `gbrs: url:` in config.yaml to a .tar.gz of the bundle (see the Zenodo
    record linked in the README). If you already have the files on disk, point
    `gbrs: data_dir:` at them instead and this rule never runs.
    """
    output:
        BOWTIE_INDEX_FILES,
        gbrs_file("transcript_info"),
        gbrs_file("gene2transcripts"),
        gbrs_file("transcript_lengths"),
        gbrs_file("gene_pos"),
        gbrs_file("genome_grid"),
        gbrs_file("emissions"),
    params:
        url = GBRS["url"],
        dir = GBRS_DIR,
        bowtie_index_tar = GBRS["bowtie_index_tar"],
        trans_info = GBRS["transcript_info"],
        gene2trans = GBRS["gene2transcripts"],
        trans_length = GBRS["transcript_lengths"],
        gene_pos = GBRS["gene_pos"],
        genome_grid = GBRS["genome_grid"],
        emissions = GBRS["emissions"],
    resources:
        runtime = '8h',
    container:
        "images/gbrs.sif"
    shell:
        """
        if [ "{params.url}" = "None" ]; then
            echo "ERROR: the GBRS reference files are missing and no download URL is set." >&2
            echo "Either point gbrs.data_dir in config.yaml at an existing copy of the" >&2
            echo "supporting-files bundle, or set gbrs.url to a .tar.gz to download." >&2
            exit 1
        fi
        mkdir -p {params.dir}
        curl -L --fail --retry 3 -o {params.dir}/{params.trans_info} "{params.url}/files/{params.trans_info}"
        curl -L --fail --retry 3 -o {params.dir}/{params.gene2trans} "{params.url}/files/{params.gene2trans}"
        curl -L --fail --retry 3 -o {params.dir}/{params.trans_length} "{params.url}/files/{params.trans_length}"
        curl -L --fail --retry 3 -o {params.dir}/{params.gene_pos} "{params.url}/files/{params.gene_pos}"
        curl -L --fail --retry 3 -o {params.dir}/{params.genome_grid} "{params.url}/files/{params.genome_grid}"
        curl -L --fail --retry 3 -o {params.dir}/{params.emissions} "{params.url}/files/{params.emissions}"

        curl -L --fail --retry 3 -o {params.dir}/{params.bowtie_index_tar} "{params.url}/files/{params.bowtie_index_tar}"
        cd {params.dir}
        tar -xzf {params.bowtie_index_tar} --strip-components 1
        """
