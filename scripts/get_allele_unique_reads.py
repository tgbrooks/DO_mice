import pathlib
import yaml
import numpy as np
import polars as pl
import polars_bio as pb
import scipy.sparse
from util.compressed_emase import load_compressed_emase

cfg = yaml.load(pathlib.Path("config.yaml").open(), yaml.Loader)
haplotype_names = cfg["haplotypes"].split(",")

sample_id = "DO024"
H5 = snakemake.input.h5
haplotypes = [f"h{i}" for i in range(8)]  # compressed.h5 labels them h0, ..., h7
data = load_compressed_emase(H5, haplotypes)

genotypes = pl.read_csv(snakemake.input.genotypes, separator="\t").rename(
    {"#Gene_ID": "gene_id", "Diplotype": "diplotype"}
)
gene_ids = genotypes["gene_id"]

annot = (
    pb.scan_gtf(
        cfg["gtf"],
        attr_fields=["gene_id", "transcript_id", "transcript_version"],
    )
    .filter(pl.col("type") == "transcript")
    .collect()
)

transcript_genotypes = genotypes.join(
    annot.select("gene_id", "transcript_id"), "gene_id", how="left"
)

# Make sure the reference matches
assert not transcript_genotypes["transcript_id"].is_null().any(), (
    "all genotyped genes should be annotated"
)
assert np.isin(data.lname, annot["transcript_id"]).all(), (
    "all quantified transcripts should be annotated"
)
# NOTE: some quantified genes are not genotyped
# they will get masked out entirely


# Group the haplotype compatibility by GENE instead of transcript
# A read is compatible with a gene if it is compatible with any of its transcripts
def group_by(compat):
    # We do this by hand since no built-in group-by for sparse arrays
    # Convert sparse matrix to dataframe containing the indices
    conv = pl.DataFrame(
        {
            "transcript_id": data.lname.astype(str),
            "indptr": compat.indptr[:-1],
            "indices": [
                compat.indices[compat.indptr[i] : compat.indptr[i + 1]]
                for i in range(len(compat.indptr) - 1)
            ],
        }
    )
    # Perform the groupby "any" column
    # which is just taking any indices that occur in any of the transcripts:
    # that column (read group) is compatible with the gene
    grouped = (
        conv.join(annot.select("transcript_id", "gene_id"), "transcript_id", how="left")
        .group_by("gene_id")
        .agg(pl.col("indices").explode(empty_as_null=False))
        .select("gene_id", indices=pl.col("indices").list.unique())
    )
    # force the ordering to be consistent
    grouped = genotypes.select("gene_id").join(grouped, "gene_id", "left")

    # convert back to sparse
    indices = grouped.select(pl.col("indices").explode(empty_as_null=False))["indices"]
    indptr = np.concat(
        (
            grouped.select(pl.col("indices").list.len().cum_sum())["indices"],
            [len(indices)],
        )
    )
    all_ones = np.ones(len(indices))
    sparse = scipy.sparse.csr_matrix(
        (all_ones, indices, indptr), shape=(genotypes.shape[0], compat.shape[1])
    )
    return sparse


print("Grouping compatibility matrices by gene")
gene_haps = {hap: group_by(compat) for hap, compat in data.haps.items()}

# Mask for just the founder alleles that are consistent with the diploid genotype
diploid_mask = {
    hap: genotypes["diplotype"].str.contains(hap_name).fill_null(False).to_numpy()
    for hap, hap_name in zip(haplotypes, haplotype_names)
}

# we only care about compatibility with measured diploitype of each gene
compatible = {
    hap: gene_haps[hap].multiply(diploid_mask[hap][:, None]) for hap in haplotypes
}

# Count how many haplotypes it could be compatible with
num_hap_compat = sum(compatible.values())
uniquely_compatible_groups = scipy.sparse.coo_matrix(num_hap_compat == 1)
uniquely_compatible_reads = uniquely_compatible_groups.multiply(data.count)

# Determine how many reads are allele-specific for each gene
unique_count_by_gene = np.asarray(uniquely_compatible_reads.sum(axis=1))[:, 0]

# NOTE: one complexity here is that a read could be compatible with multiple gene
# with different genotypes. I.e., it could be compatible with founder A
# in gene 1 and founder B in gene 2. We only look at ambiguity within a gene.

df = pl.DataFrame(
    {
        "gene_id": gene_ids,
        "allele_specific_reads": unique_count_by_gene.astype(int),
    }
)

df.write_parquet(snakemake.output.gene_unique)
