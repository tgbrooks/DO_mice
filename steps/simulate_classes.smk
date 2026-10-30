SIM_REF_TISSUE = "Adipose"
SIM_MOUSE_IDS = [mouse for mouse in MICE[SIM_REF_TISSUE] if mouse not in config['outlier_ids']]
rule simulate_classes:
    """ Create compressed.h5 class-level read counts for samples """
    input:
        gene_ids = "resources/simulated_gene_ids.txt",
        sf = f"processed/{SIM_REF_TISSUE}/size_factors.txt",
        eff_len = f"processed/{SIM_REF_TISSUE}/effective_lengths.by_gene.parquet",
        allele_unique = f"processed/{SIM_REF_TISSUE}/allele_unique_reads.parquet",
        genotypes = "processed/genotypes.parquet",
        pheontypes = "phenotypes.csv.gz",
        buffering_results = f"results/{SIM_REF_TISSUE}/buffering_with_bias.ndjson",
        emase = expand("processed/{SIM_REF_TISSUE}/gbrs/{mouse_id}.compressed.h5", SIM_REF_TISSUE=[SIM_REF_TISSUE], mouse_id=SIM_MOUSE_IDS), # a consequence of allele_unique and too messy to include
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

