import numpy as np
import polars as pl
import scipy.sparse
from util.compressed_emase import Emase


# Group the haplotype compatibility by GENE instead of transcript
# A read is compatible with a gene if it is compatible with any of its transcripts
def group_by(compat, transcript_ids, annot, genotypes):
    # We do this by hand since no built-in group-by for sparse arrays
    # Convert sparse matrix to dataframe containing the indices
    conv = pl.DataFrame(
        {
            "transcript_id": transcript_ids,
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
    grouped = (
        genotypes.select("gene_id")
        .join(grouped, "gene_id", "left")
        .with_columns(indices=pl.col("indices").fill_null([]))
    )

    # convert back to sparse
    indices = grouped.select(pl.col("indices").explode(empty_as_null=False))["indices"]
    indptr = np.concat(
        (
            [0],
            grouped.select(pl.col("indices").list.len().cum_sum())["indices"],
        )
    )
    all_ones = np.ones(len(indices))
    sparse = scipy.sparse.csr_matrix(
        (all_ones, indices, indptr), shape=(genotypes.shape[0], compat.shape[1])
    )
    return sparse


def summarize_count_types(data, annot, genotypes):
    """compute total, allele-specific, and diplotype incompatible reads"""
    haplotypes = data.haps.keys()
    haplotype_names = data.hname
    gene_haps = {
        hap: group_by(compat, data.lname.astype(str), annot, genotypes)
        for hap, compat in data.haps.items()
    }

    # Mask for just the founder alleles that are consistent with the diploid genotype
    diploid_mask = {
        hap: genotypes["diplotype"].str.contains(hap_name).fill_null(False).to_numpy()
        for hap, hap_name in zip(haplotypes, haplotype_names)
    }

    # we only care about compatibility with measured diplotype of each gene
    compatible = {
        hap: gene_haps[hap].multiply(diploid_mask[hap][:, None]) for hap in haplotypes
    }

    # Count how many haplotypes it could be compatible with
    num_hap_compat = sum(compatible.values())
    uniquely_compatible_groups = scipy.sparse.coo_matrix(num_hap_compat == 1)
    uniquely_compatible_reads = uniquely_compatible_groups.multiply(data.count)

    # Determine how many reads are allele-specific for each gene
    unique_count_by_gene = np.asarray(uniquely_compatible_reads.sum(axis=1))[:, 0]

    #### Count how many reads are compatible only with haplotypes not in the expected diplotype
    other_compatible = {
        hap: gene_haps[hap].multiply(~diploid_mask[hap][:, None]) for hap in haplotypes
    }
    num_other_compat = sum(other_compatible.values())
    only_other_compat_groups = (
        (num_other_compat > 0) - (num_hap_compat > 0).multiply(num_other_compat > 0)
    ) > 0
    only_other_compat_reads = np.asarray(
        np.sum(only_other_compat_groups.multiply(data.count), axis=1)
    )[:, 0]

    #### Count reads unique to each allele
    haplotype1_mask = {
        hap: genotypes["diplotype"]
        .str.starts_with(hap_name)
        .fill_null(False)
        .to_numpy()
        for hap, hap_name in zip(haplotypes, haplotype_names)
    }
    haplotype2_mask = {
        hap: genotypes["diplotype"].str.ends_with(hap_name).fill_null(False).to_numpy()
        for hap, hap_name in zip(haplotypes, haplotype_names)
    }
    hap1_compat = sum(
        gene_haps[hap].multiply(haplotype1_mask[hap][:, None]) for hap in haplotypes
    )
    hap2_compat = sum(
        gene_haps[hap].multiply(haplotype2_mask[hap][:, None]) for hap in haplotypes
    )
    hap1_unique_groups = (hap1_compat - hap2_compat) > 0
    hap1_unique_reads = np.asarray(
        np.sum(hap1_unique_groups.multiply(data.count), axis=1)
    )[:, 0].astype(int)
    hap2_unique_groups = (hap2_compat - hap1_compat) > 0
    hap2_unique_reads = np.asarray(
        np.sum(hap2_unique_groups.multiply(data.count), axis=1)
    )[:, 0].astype(int)
    # NOTE: one complexity here is that a read could be compatible with multiple gene
    # with different genotypes. I.e., it could be compatible with founder A
    # in gene 1 and founder B in gene 2. We only look at ambiguity within a gene.

    ##### Count total reads compatible with a gene
    compat_groups = sum(gene_haps.values()) > 0
    compat_reads = compat_groups.multiply(data.count)
    total_reads = np.asarray(np.sum(compat_reads, axis=1))[:, 0].astype(int)

    def num_incompat_with_hap(hap):
        # Count reads compatible with *some* haplotype of the gene but not with `hap`
        incompat_groups = (compat_groups - gene_haps[hap]) > 0
        incompat_reads = incompat_groups.multiply(data.count)
        return np.asarray(np.sum(incompat_reads, axis=1))[:, 0].astype(int)

    df = pl.DataFrame(
        {
            "gene_id": genotypes["gene_id"],
            "total_reads": total_reads,
            "allele_specific_reads": unique_count_by_gene.astype(int),
            "diplotype": genotypes["diplotype"],
            # NOTE: these are null if homozygous
            "haplotype_1_unique": hap1_unique_reads,
            "haplotype_2_unique": hap2_unique_reads,
            "diplotype_incompat_reads": only_other_compat_reads,
            **{
                f"not_{hap_name}_reads": num_incompat_with_hap(hap)
                for hap_name, hap in zip(haplotype_names, haplotypes)
            },
        }
    )
    return df


# Perform a test of the summarize function
test_data = Emase(
    shape=(1, 3, 7),
    count=np.array([1, 2, 3, 4, 5, 6, 7]),
    haps=dict(
        h0=scipy.sparse.csr_matrix(np.array([[1, 0, 0, 1, 1, 0, 1]])),
        h1=scipy.sparse.csr_matrix(np.array([[0, 1, 0, 1, 1, 0, 0]])),
        h2=scipy.sparse.csr_matrix(np.array([[0, 0, 1, 1, 0, 0, 1]])),
    ),
    lname=np.array(["transcript1"]),
    hname=["A", "B", "C"],
)
test_genotypes = pl.DataFrame({"gene_id": ["gene1"], "diplotype": ["AB"]})
test_annot = pl.DataFrame({"gene_id": ["gene1"], "transcript_id": ["transcript1"]})
out = summarize_count_types(test_data, test_annot, test_genotypes)
assert out.to_dicts() == [
    {
        "gene_id": "gene1",
        "total_reads": 22,
        "allele_specific_reads": 10,
        "diplotype": "AB",
        "haplotype_1_unique": 8,
        "haplotype_2_unique": 2,
        "diplotype_incompat_reads": 3,
        "not_A_reads": 5,
        "not_B_reads": 11,
        "not_C_reads": 8,
    }
]
