"""Genotypes: array-based founder haplotype probabilities for the DO mice.

The genotypes are distributed as an .RData file holding a `genoprobs` list with
one 3-D array per chromosome (mice x 8 founders x markers), mice labelled like
`DO021` as in GEO. They are turned into the per-gene diplotype calls that
`gbrs quantify` accepts, so expression can be quantified against the known
genome of each mouse instead of one reconstructed from the RNA-seq itself.

    geno/genoprobs.RData                    downloaded
    geno/genotyped_mice.txt                 mouse IDs present in the file
    geno/alleleprobs/{mouse}.tsv.gz         per-mouse founder probabilities
    geno/gbrs_genotypes/{mouse}.genotypes.tsv   per-gene diplotype calls for GBRS
"""

GENO = config["genotypes"]


rule download_genoprobs:
    """Download the .RData file of founder haplotype probabilities."""
    output:
        "geno/genoprobs.RData",
    params:
        url = GENO["url"],
    resources:
        runtime = '4h',
    container:
        "images/rgeno.sif"
    shell:
        """
        mkdir -p geno
        curl -L --fail --retry 3 -o {output} "{params.url}"
        """


rule list_genotyped_mice:
    """List the mouse IDs in the genotype file, one per line.

    Generate this before the main run (`snakemake genotyped_mice`) so that
    sample selection can skip RNA-seq samples that have no genotype.
    """
    input:
        rdata = "geno/genoprobs.RData",
    output:
        GENOTYPED_MICE,
    params:
        object = GENO["object"],
    resources:
        mem_mb = 32000,
        runtime = '2h',
    container:
        "images/rgeno.sif"
    shell:
        """
        Rscript scripts/export_genoprobs.R \
            --rdata {input.rdata} \
            --object {params.object} \
            --list-mice {output}
        """


rule genotyped_mice:
    """Convenience target for the mouse list (see list_genotyped_mice)."""
    input:
        GENOTYPED_MICE,


rule export_alleleprobs:
    """Extract the selected mice from the .RData file as per-mouse TSVs.

    One rule for all mice, because reading the whole genoprobs object is the
    expensive part. Each output has columns `marker`, `chr`, and one column per
    founder haplotype (in the order given by config `haplotypes`).
    """
    input:
        rdata = "geno/genoprobs.RData",
    output:
        probs = expand("geno/alleleprobs/{mouse}.tsv.gz", mouse=ALL_MICE),
        markers = "geno/markers.tsv",
        founders = "geno/founder_order.txt",
    params:
        object = GENO["object"],
        mice = ",".join(ALL_MICE),
        outdir = "geno/alleleprobs",
        haplotypes = HAPLOTYPES,
    resources:
        mem_mb = 32000,
        runtime = '4h',
    container:
        "images/rgeno.sif"
    shell:
        """
        Rscript scripts/export_genoprobs.R \
            --rdata {input.rdata} \
            --object {params.object} \
            --mice {params.mice} \
            --haplotypes {params.haplotypes} \
            --outdir {params.outdir} \
            --markers {output.markers} \
            --founder-order {output.founders}
        """


rule genoprobs_to_gbrs:
    """Convert a mouse's founder probabilities into GBRS per-gene diplotypes.

    Each gene is assigned the founder probabilities of its nearest genotyped
    marker, which are then called as a homozygous or heterozygous diplotype
    (e.g. `AA`, `CF`) in the format `gbrs quantify -G` expects.
    """
    input:
        probs = "geno/alleleprobs/{mouse}.tsv.gz",
        markers = "geno/markers.tsv",
        grid = gbrs_file("genome_grid"),
        gene_pos = gbrs_file("gene_pos"),
        gene2transcripts = gbrs_file("gene2transcripts"),
    output:
        "geno/gbrs_genotypes/{mouse}.genotypes.tsv",
    params:
        haplotypes = HAPLOTYPES,
        hom_threshold = GENO["hom_dosage_threshold"],
        min_call_prob = GENO["min_call_prob"],
        marker_units = GENO["marker_units"],
    resources:
        mem_mb = 8000,
        runtime = '1h',
    container:
        "images/gbrs.sif"
    shell:
        """
        python3 scripts/genoprobs_to_gbrs.py \
            --alleleprobs {input.probs} \
            --markers {input.markers} \
            --grid {input.grid} \
            --gene-pos {input.gene_pos} \
            --gene2transcripts {input.gene2transcripts} \
            --haplotypes {params.haplotypes} \
            --hom-dosage-threshold {params.hom_threshold} \
            --min-call-prob {params.min_call_prob} \
            --marker-units {params.marker_units} \
            --sample {wildcards.mouse} \
            --out {output}
        """
