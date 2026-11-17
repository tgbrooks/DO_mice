SIM_REF_TISSUE = "Adipose"
SIM_MOUSE_IDS = [mouse for mouse in MICE[SIM_REF_TISSUE] if mouse not in config['outlier_ids'][SIM_REF_TISSUE]]
rule simulate_classes:
    """ Create compressed.h5 class-level read counts for samples """
    input:
        gene_ids = "resources/simulated_gene_ids.txt",
        sf = f"processed/{SIM_REF_TISSUE}/size_factors.txt",
        eff_len = f"processed/{SIM_REF_TISSUE}/effective_lengths.by_gene.parquet",
        allele_unique = f"processed/{SIM_REF_TISSUE}/allele_unique_reads.parquet",
        genotypes = "processed/genotypes.parquet",
        pheontypes = "phenotypes.csv.gz",
        buffering_results = f"results/{SIM_REF_TISSUE}/buffering_full_results.parquet",
        #emase = expand(f"processed/{SIM_REF_TISSUE}/gbrs/{mouse_id}.compressed.h5", SIM_REF_TISSUE=[SIM_REF_TISSUE], mouse_id=SIM_MOUSE_IDS), # a consequence of allele_unique and too messy to include
    output:
        "processed/simulated_classes/gene_truths.parquet",
        "processed/simulated_classes/sample_truths.parquet",
        expand("processed/simulated_classes/gbrs/{mouse_id}.compressed.h5", mouse_id = SIM_MOUSE_IDS),
    params:
        outdir ="processed/simulated_classes/",
    resources:
            mem_mb = 24_000,
    shell:
        f"python scripts/simulate_classes.py --tissue {SIM_REF_TISSUE} --gene_id_file {{input.gene_ids}} --outdir {{params.outdir}}"

rule combine_allele_unique_reads_simulated_classes:
    input:
        allele_unique = lambda w: expand("processed/simulated_classes/gbrs_allele_unique_reads/{mouse_id}.allele_unique_reads.parquet",
            mouse_id = SIM_MOUSE_IDS,
        ),
    output:
        allele_unique = "processed/simulated_classes/allele_unique_reads.parquet"
    script:
        "../scripts/combine_allele_unique_reads.py"

rule copy_from_ref:
    """ Make duplicates of some files from the reference tissue """
    input:
        f"processed/{SIM_REF_TISSUE}/size_factors.txt",
        f"processed/{SIM_REF_TISSUE}/effective_lengths.by_gene.parquet",
    output:
        "processed/simulated_classes/size_factors.txt",
        "processed/simulated_classes/effective_lengths.by_gene.parquet",
    run:
        import pathlib
        for (infile, outfile) in zip(input, output):
            data = pathlib.Path(infile).read_bytes()
            pathlib.Path(outfile).write_bytes(data)

n_chunks = config['simulated_classes']['n_chunks']
rule chunk_simulated_classes:
    input: gene_ids="resources/simulated_gene_ids.txt"
    output: chunk="processed/simulated_classes/chromosome_chunks/1.{i}.genes.txt"
    run:
        ids = [x.strip() for x in open(input.gene_ids).read().strip().splitlines()]
        i = int(wildcards.i)
        out_ids = ids[int(i)::n_chunks]
        import polars as pl
        pl.DataFrame({"gene_id": out_ids}).write_csv(output.chunk)

# ------------------------------------------------------------------------------
# Oracle runs: ASE model with the true alpha and/or q plugged in, to isolate which
# estimated input is responsible for miscalibrated ASE beta covariances.
# Run with:  snakemake ase_oracle_tests
# Only touches processed/simulated_classes/; nothing for the real tissues reruns.
# ------------------------------------------------------------------------------
ORACLE_CHUNKS = [1, 2]  # ~100 simulated genes before the expression filter

rule run_ase_model_part1_oracle:
    input:
        gene_list = "processed/simulated_classes/chromosome_chunks/1.{chunk_num}.genes.txt",
        truths = "processed/simulated_classes/sample_truths.parquet",
    output:
        "processed/simulated_classes/oracle_{oracle}/1.{chunk_num}.ase_model_part1.nc"
    wildcard_constraints:
        oracle = "none|alpha|q|both",
    threads: 4
    resources:
        mem_mb = 32_000,
    shell:
        "python scripts/run_ase_model_part1.py --tissue simulated_classes"
        " --oracle {wildcards.oracle} --ref_tissue {SIM_REF_TISSUE}"
        " --outfile {output} --gene_ids_file {input.gene_list}"

rule run_ase_model_part2_oracle:
    input:
        gene_list = "processed/simulated_classes/chromosome_chunks/1.{chunk_num}.genes.txt",
        part1 = "processed/simulated_classes/oracle_{oracle}/1.{chunk_num}.ase_model_part1.nc",
    output:
        "processed/simulated_classes/oracle_{oracle}/1.{chunk_num}.ase_model_part2.json"
    wildcard_constraints:
        oracle = "none|alpha|q|both",
    resources:
        mem_mb = 24_000,
    shell:
        "python scripts/run_ase_model_part2.py --tissue simulated_classes"
        " --outfile {output} --gene_ids_file {input.gene_list} --part1 {input.part1}"

rule ase_oracle_tests:
    input:
        part2 = expand(
            "processed/simulated_classes/oracle_{oracle}/1.{chunk_num}.ase_model_part2.json",
            oracle = ["none", "alpha", "q", "both"],
            chunk_num = ORACLE_CHUNKS,
        ),
        gene_truths = "processed/simulated_classes/gene_truths.parquet",
        sample_truths = "processed/simulated_classes/sample_truths.parquet",
    output:
        "processed/simulated_classes/oracle_tests.txt"
    shell:
        "python scripts/check_ase_oracle.py --outfile {output} {input.part2}"

# ------------------------------------------------------------------------------
# sigma_q sweep: width of the stage-1 prior on q around q_hat (default 0.3, i.e. the
# oracle_none run). Tests whether the q posterior is overconfident because the prior
# is centered on a q_hat estimated from the same counts.
# Run with:  snakemake sigma_q_sweep_tests
# ------------------------------------------------------------------------------
SIGMA_Q_SWEEP = ["1", "3", "10"]

rule run_ase_model_part1_sigma_q:
    input:
        gene_list = "processed/simulated_classes/chromosome_chunks/1.{chunk_num}.genes.txt",
    output:
        "processed/simulated_classes/sigmaq_{sigma_q}/1.{chunk_num}.ase_model_part1.nc"
    wildcard_constraints:
        sigma_q = r"[0-9]+(\.[0-9]+)?",
    threads: 4
    resources:
        mem_mb = 32_000,
    shell:
        "python scripts/run_ase_model_part1.py --tissue simulated_classes"
        " --config '{{\"sigma_q\": {wildcards.sigma_q}}}'"
        " --outfile {output} --gene_ids_file {input.gene_list}"

rule run_ase_model_part2_sigma_q:
    input:
        gene_list = "processed/simulated_classes/chromosome_chunks/1.{chunk_num}.genes.txt",
        part1 = "processed/simulated_classes/sigmaq_{sigma_q}/1.{chunk_num}.ase_model_part1.nc",
    output:
        "processed/simulated_classes/sigmaq_{sigma_q}/1.{chunk_num}.ase_model_part2.json"
    wildcard_constraints:
        sigma_q = r"[0-9]+(\.[0-9]+)?",
    resources:
        mem_mb = 24_000,
    shell:
        "python scripts/run_ase_model_part2.py --tissue simulated_classes"
        " --outfile {output} --gene_ids_file {input.gene_list} --part1 {input.part1}"

rule sigma_q_sweep_tests:
    input:
        part2 = [
            # sigma_q = 0.3 (the default) is the oracle_none run
            *expand(
                "processed/simulated_classes/oracle_none/1.{chunk_num}.ase_model_part2.json",
                chunk_num = ORACLE_CHUNKS,
            ),
            *expand(
                "processed/simulated_classes/sigmaq_{sigma_q}/1.{chunk_num}.ase_model_part2.json",
                sigma_q = SIGMA_Q_SWEEP,
                chunk_num = ORACLE_CHUNKS,
            ),
        ],
        gene_truths = "processed/simulated_classes/gene_truths.parquet",
        sample_truths = "processed/simulated_classes/sample_truths.parquet",
    output:
        "processed/simulated_classes/sigma_q_sweep_tests.txt"
    shell:
        "python scripts/check_ase_oracle.py --outfile {output} {input.part2}"

# ------------------------------------------------------------------------------
# eta prior sweep (at sigma_q = 1): stage 1 puts an independent N(offset, tau^2) prior
# on each mouse's eta, divided out again in part 2. With tau = 3 (the default) shared q
# uncertainty can make the posterior nearly as wide as the prior along directions
# shared by many mice, which breaks the division. A wider tau avoids that.
# tau = 3 at sigma_q = 1 is the sigmaq_1 run. Part 2 reads tau from the part-1 file.
# Run with:  snakemake eta_prior_sweep_tests
# ------------------------------------------------------------------------------
ETA_PRIOR_SWEEP = ["10", "30"]

rule run_ase_model_part1_eta_prior:
    input:
        gene_list = "processed/simulated_classes/chromosome_chunks/1.{chunk_num}.genes.txt",
    output:
        "processed/simulated_classes/etasd_{eta_sd}/1.{chunk_num}.ase_model_part1.nc"
    wildcard_constraints:
        eta_sd = r"[0-9]+(\.[0-9]+)?",
    threads: 4
    resources:
        mem_mb = 32_000,
    shell:
        "python scripts/run_ase_model_part1.py --tissue simulated_classes"
        " --config '{{\"sigma_q\": 1, \"eta_prior_sd\": {wildcards.eta_sd}}}'"
        " --outfile {output} --gene_ids_file {input.gene_list}"

rule run_ase_model_part2_eta_prior:
    input:
        gene_list = "processed/simulated_classes/chromosome_chunks/1.{chunk_num}.genes.txt",
        part1 = "processed/simulated_classes/etasd_{eta_sd}/1.{chunk_num}.ase_model_part1.nc",
    output:
        "processed/simulated_classes/etasd_{eta_sd}/1.{chunk_num}.ase_model_part2.json"
    wildcard_constraints:
        eta_sd = r"[0-9]+(\.[0-9]+)?",
    resources:
        mem_mb = 24_000,
    shell:
        "python scripts/run_ase_model_part2.py --tissue simulated_classes"
        " --outfile {output} --gene_ids_file {input.gene_list} --part1 {input.part1}"

rule eta_prior_sweep_tests:
    input:
        part2 = [
            *expand(
                "processed/simulated_classes/sigmaq_1/1.{chunk_num}.ase_model_part2.json",
                chunk_num = ORACLE_CHUNKS,
            ),
            *expand(
                "processed/simulated_classes/etasd_{eta_sd}/1.{chunk_num}.ase_model_part2.json",
                eta_sd = ETA_PRIOR_SWEEP,
                chunk_num = ORACLE_CHUNKS,
            ),
        ],
        gene_truths = "processed/simulated_classes/gene_truths.parquet",
        sample_truths = "processed/simulated_classes/sample_truths.parquet",
    output:
        "processed/simulated_classes/eta_prior_sweep_tests.txt"
    shell:
        "python scripts/check_ase_oracle.py --outfile {output} {input.part2}"
