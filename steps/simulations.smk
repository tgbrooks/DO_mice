import math
N_NO_CIS_GENES = 200
N_NO_BUFFERING_GENES = 200
N_BUFFERING_GENES = 200
N_GENES = N_NO_CIS_GENES + N_NO_BUFFERING_GENES + N_BUFFERING_GENES

rule simulate_counts:
    """ Simulate count-level data for our buffering model """
    output:
        "processed/simulated_counts/kinship.txt",
        "processed/simulated_counts/genotypes.parquet",
        "processed/simulated_counts/phenotypes.csv",
        "processed/simulated_counts/gene_annot.txt",
        "processed/simulated_counts/size_factors.txt",
        "processed/simulated_counts/simulated_counts.diploid.genes.founder_expected_read_counts.parquet",
        "processed/simulated_counts/allele_unique_reads.parquet",
        "processed/simulated_counts/true_params.txt",
    params:
        N_SAMPLES = 200,
        N_NO_CIS_GENES = N_NO_CIS_GENES ,
        N_NO_BUFFERING_GENES = N_NO_BUFFERING_GENES ,
        N_BUFFERING_GENES = N_BUFFERING_GENES ,
        SEED = 100,
    script:
        "../scripts/simulate_counts.py"

CHUNK_SIZE = 50
rule model_buffering_simulated_counts:
    input:
        counts = "processed/simulated_counts/simulated_counts.diploid.genes.founder_expected_read_counts.parquet",
        size_factors = "processed/simulated_counts/size_factors.txt",
        phenotypes = "processed/simulated_counts/phenotypes.csv",
        genotypes = "processed/simulated_counts/genotypes.parquet",
        kinship = "processed/simulated_counts/kinship.txt",
        allele_unique_reads = "processed/simulated_counts/allele_unique_reads.parquet",
    output:
        outfile = "processed/simulated_counts/buffering/{chromosome}.{chunk_num}.txt", # only chromsome 1 is actually present
    params:
        min_median_counts = config['MIN_MEDIAN_COUNTS'],
        genes = lambda wildcards: [f'GENE{i:04}' for i  in range(int(wildcards.chunk_num)*CHUNK_SIZE, (int(wildcards.chunk_num)+1)*CHUNK_SIZE)]
    resources:
        mem_mb = 18_000,
    container:
        "images/rgeneral.sif"
    script:
        "../scripts/model_buffering.R"


rule check_model_on_simulations:
    input:
        results =  expand("processed/simulated_counts/buffering/1.{chunk_num}.txt", chunk_num=range(math.ceil(N_GENES / CHUNK_SIZE))),
        truth = "processed/simulated_counts/true_params.txt",
    output:
        report = "processed/simulated_counts/report.txt"
    shell:
        "python ../scripts/check_model_on_simulations.py > {output}"

