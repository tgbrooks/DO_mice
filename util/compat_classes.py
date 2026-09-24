from dataclasses import dataclass
import polars as pl
import scipy.sparse
import numpy as np


@dataclass
class CompatClasses:
    compat: (
        np.ndarray
    )  # 8 x n_classes - compatibility of the read class with each of the 8 haplotypes
    counts: np.ndarray  # n_classes - number of reads of this compatibility class
    haplotypes: list[str]  # n_haplotpyes - names of haplotypes (e.g., "A", "B", ...)


def sparse_any(sparse_matrix, axis):
    return (sparse_matrix != 0).sum(axis=axis) > 0


@dataclass
class CompatClassesDf:
    """All read compatibility data from one gene across all samples"""

    compat: np.ndarray  # 8 x n_classes
    counts: np.ndarray  # n_samples x n_classes
    ids: list[str]  # n_samples ids list
    haplotypes: list[str]  # n_haplotypes


# Now uniformize it: all samples to have the *same* compatibility classes
def uniformize_compat_classes(
    compat_classes: dict[str, CompatClasses], min_n_classes: int | None
) -> CompatClassesDf:
    """
    Aggregate classes and counts, putting zeros in classes that are missing for any sample.
    Pad classes to min_n_classes.
    """
    n_samples = len(compat_classes)
    assert n_samples > 0
    classes = set()
    for compat in compat_classes.values():
        classes.update([tuple(col) for col in compat.compat.T])
    classes = np.array([np.array(cls) for cls in classes])  # fix the ordering
    if min_n_classes is not None and classes.shape[0] < min_n_classes:
        # Pad with classes that incompatible with all haplotypes
        # These should have no reads reported since such reads simply don't align to this gene
        n_padding_classes = min_n_classes - len(classes)
        classes = np.concatenate(
            [
                classes,
                np.zeros((n_padding_classes, classes.shape[1]), dtype=bool),
            ]
        )
    compat_map = {tuple(row): i for i, row in enumerate(classes)}
    counts_out = np.zeros((n_samples, classes.shape[0]))
    for i, compat in enumerate(compat_classes.values()):
        matches = np.array(
            [compat_map[tuple(col)] for col in compat.compat.T], dtype=int
        )
        # matches may have repeated indices due to transcripts classes being identical at gene level
        # we sum those repeated indices
        np.add.at(counts_out[i], matches, compat.counts)
        # counts_out[i, matches] += compat.counts
    return CompatClassesDf(
        compat=classes,
        counts=counts_out,
        ids=list(compat_classes.keys()),
        haplotypes=compat.haplotypes,
    )


def extract_gene(emase, transcripts):
    """get data for just one gene from an emase file
    aggregating over all the transcripts"""
    tx = np.isin(emase.lname, [tx.encode() for tx in transcripts])
    # 8 x n_classes array of compatibility with this gene
    gene_compat = np.array(
        [
            sparse_any(scipy.sparse.csr_array(compat[tx, :]), axis=0)
            for compat in emase.haps.values()
        ]
    )
    relevant_classes = sparse_any(gene_compat, axis=0)
    final_compat = gene_compat[:, relevant_classes]
    counts = emase.count[relevant_classes]
    # note: classes arise from *transcripts* not genes and therefore could have identical gene compatibility
    # in different classes. we'll combine those in the next step.
    return CompatClasses(compat=final_compat, counts=counts, haplotypes=emase.hname)


def get_gene_class_counts(
    gene_id, tx_annot, all_compatibility_classes
) -> CompatClassesDf:
    """Get class counts for all samples for a specific gene"""
    # select just this genes data
    transcripts = list(tx_annot.filter(gene_id=gene_id)["transcript_id"])

    compatibility_classes = {
        mouse_id: extract_gene(emase, transcripts)
        for mouse_id, emase in all_compatibility_classes.items()
    }
    return uniformize_compat_classes(compatibility_classes, min_n_classes=None)


def get_gene_totals(
    gene_id: str, gene_class_counts: CompatClassesDf, diplotypes: pl.DataFrame
) -> pl.DataFrame:
    """Return total read counts of the gene by sample"""
    return pl.DataFrame(
        {
            "mouse_id": gene_class_counts.ids,
            "total_reads": gene_class_counts.counts.sum(axis=1),
        }
    ).join(
        diplotypes.filter(gene_id=gene_id),
        "mouse_id",
    )
