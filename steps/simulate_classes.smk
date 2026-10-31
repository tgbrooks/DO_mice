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

    #rule get_allele_unique_reads_simulated_classes:
    #    """ Count how many reads distinguish allele-specificity in each gene """
    #    input:
    #        h5 = "processed/simulated_classes/gbrs/{mouse}.compressed.h5",
    #        genotypes = "geno/gbrs_genotypes/{mouse}.genotypes.tsv",
    #        gtf = "gbrs_ref/v116/reference.gtf.gz",
    #    output:
    #        gene_unique = "processed/simulated_classes/gbrs_allele_unique_reads/{mouse}.allele_unique_reads.parquet",
    #    script:
    #        "../scripts/get_allele_unique_reads.py"

rule combine_allele_unique_reads_simulated_classes:
    input:
        allele_unique = lambda w: expand("processed/simulated_classes/gbrs_allele_unique_reads/{mouse_id}.allele_unique_reads.parquet",
            mouse_id = SIM_MOUSE_IDS,
        ),
    output:
        allele_unique = "processed/simulated_classes/allele_unique_reads.parquet"
    script:
        "../scripts/combine_allele_unique_reads.py"

n_chunks = 10
rule chunk_simulated_classes:
    input: gene_ids="resources/simulated_gene_ids.txt"
    output: chunk="processed/simulated_classes/chromosome_chunks/{i}.genes.txt"
    run:
        ids = [x.strip() for x in open(input.gene_ids).read().strip().splitlines()]
        i = int(wildcards.i)
        out_ids = ids[int(i)::n_chunks]
        import polars as pl
        pl.DataFrame({"gene_id": out_ids}).write_csv(output.chunk)

rule model_buffering_with_bias_simulated_classes:
    """ Run bias-correcting ASE models and total count models """
    input:
        allele_unique_reads = "processed/simulated_classes/allele_unique_reads.parquet",
        size_factors = f"processed/{SIM_REF_TISSUE}/size_factors.txt",
        phenotypes = "phenotypes.csv.gz",
        effective_lengths = f"processed/{SIM_REF_TISSUE}/effective_lengths.by_gene.parquet",
        gene_list = "processed/simulated_classes/chromosome_chunks/{i}.genes.txt"
    output:
        outfile = "processed/simulated_classes/buffering_with_bias/{i}.txt"
    threads: 4
    resources:
        mem_mb = 24_000,
    shell:
        "python scripts/model_buffering_2stage.py --tissue {wildcards.tissue_real} --outfile {output} --gene_ids_file {input.gene_list} --ase_config \"{{}}\""


rule collect_buffering_with_bias_results_simulated_classes:
    input:
        results = expand("processed/simulated_classes/buffering_with_bias/{i}.txt", i=range(n_chunks))
    output:
        results = "results/simulated_classes/buffering_with_bias.ndjson"
    resources:
        mem_mb = 6_000
    run:
        import json
        with open(output.results, "wt") as ndjson:
            for r in input.results:
                try:
                    data = json.load(open(r))
                except FileNotFoundError:
                    print(f"Skipping {r} - file not found")
                    continue
                for gene in data:
                    json.dump(gene, ndjson)
                    ndjson.write("\n")

