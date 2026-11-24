SIM_REF_TISSUE = "Adipose"
SIM_MOUSE_IDS = [mouse for mouse in MICE[SIM_REF_TISSUE] if mouse not in config['outlier_ids'][SIM_REF_TISSUE]]
MICE['simulated_classes'] = SIM_MOUSE_IDS
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
        out_ids = ids[i - 1::n_chunks]  # chunks are numbered 1..n_chunks
        import polars as pl
        pl.DataFrame({"gene_id": out_ids}).write_csv(output.chunk)

# ------------------------------------------------------------------------------
# Oracle runs: ASE model with the true alpha and/or q plugged in, to isolate which
# estimated input is responsible for miscalibrated ASE beta covariances.
# Run with:  snakemake ase_oracle_tests
# Only touches processed/simulated_classes/; nothing for the real tissues reruns.
# ------------------------------------------------------------------------------
ORACLE_CHUNKS = [1, 2, 3, 4, 5, 6]  # ~100 simulated genes before the expression filter

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
        " --oracle {wildcards.oracle}"
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

# ------------------------------------------------------------------------------
# Joint ASE model (ase_buffering/ase_model_joint.py), one pass, on all simulated
# chunks, with oracle variants (true alpha and/or q plugged in). Compared against the
# two-stage model (sigma_q = 1) and its oracle-q run, which cover ORACLE_CHUNKS only
# (the summary's *_common columns restrict to genes every run has).
# Sampling (.nc) and summarizing (.json) are separate steps, so summaries can be
# changed without re-sampling.
# Run with:  snakemake ase_joint_tests
# ------------------------------------------------------------------------------
import json as _json

# label -> sampler settings. Add variants here.
_JOINT_BASE = dict(config={"sigma_q": 1, "target_accept": 0.9}, draws=1000)
JOINT_RUNS = {
    "joint_ta09_d1000": dict(_JOINT_BASE, oracle="none"),
    "joint_oracle_alpha": dict(_JOINT_BASE, oracle="alpha"),
    "joint_oracle_q": dict(_JOINT_BASE, oracle="q"),
    "joint_oracle_both": dict(_JOINT_BASE, oracle="both"),
}
JOINT_CHUNKS = list(range(1, n_chunks + 1))  # all simulated genes

rule run_ase_model_joint:
    input:
        gene_list = "processed/simulated_classes/chromosome_chunks/1.{chunk_num}.genes.txt",
    output:
        "processed/simulated_classes/{joint_run}/1.{chunk_num}.ase_model_joint.nc"
    wildcard_constraints:
        joint_run = "|".join(JOINT_RUNS),
    params:
        config = lambda w: _json.dumps(JOINT_RUNS[w.joint_run]["config"]),
        draws = lambda w: JOINT_RUNS[w.joint_run]["draws"],
        oracle = lambda w: JOINT_RUNS[w.joint_run]["oracle"],
    threads: 4
    resources:
        mem_mb = 32_000,
    shell:
        "python scripts/run_ase_model_joint.py --tissue simulated_classes"
        " --config '{params.config}' --draws {params.draws}"
        " --oracle {params.oracle}"
        " --outfile {output} --gene_ids_file {input.gene_list}"

rule summarize_ase_model_joint:
    input:
        samples = "processed/simulated_classes/{joint_run}/1.{chunk_num}.ase_model_joint.nc",
        gene_list = "processed/simulated_classes/chromosome_chunks/1.{chunk_num}.genes.txt",
    output:
        "processed/simulated_classes/{joint_run}/1.{chunk_num}.ase_model_joint.json"
    wildcard_constraints:
        joint_run = "|".join(JOINT_RUNS),
    resources:
        mem_mb = 16_000,
    shell:
        "python scripts/summarize_ase_model_joint.py --samples {input.samples}"
        " --gene_ids_file {input.gene_list} --outfile {output}"

rule ase_joint_tests:
    input:
        joint = expand(
            "processed/simulated_classes/{joint_run}/1.{chunk_num}.ase_model_joint.json",
            joint_run = list(JOINT_RUNS),
            chunk_num = JOINT_CHUNKS,
        ),
        two_stage = expand(
            "processed/simulated_classes/{run}/1.{chunk_num}.ase_model_part2.json",
            run = ["sigmaq_1", "oracle_q"],
            chunk_num = ORACLE_CHUNKS,
        ),
        gene_truths = "processed/simulated_classes/gene_truths.parquet",
    output:
        "processed/simulated_classes/ase_joint_tests.txt"
    shell:
        "python scripts/summarize_ase_joint.py --outfile {output} {input.joint} {input.two_stage}"

# ------------------------------------------------------------------------------
# End-to-end check of the buffering pipeline (joint ASE model + total model +
# Deming regression) on the simulated data: lambda estimates and tests vs the truth.
# Run with:  snakemake sim_buffering_tests
# ------------------------------------------------------------------------------
rule sim_buffering_tests:
    input:
        results = "results/simulated_classes/buffering_full_results.parquet",
        truths = "processed/simulated_classes/gene_truths.parquet",
    output:
        "processed/simulated_classes/buffering_tests.txt"
    shell:
        "python scripts/summarize_buffering_sim.py --results {input.results}"
        " --truths {input.truths} --outfile {output}"
