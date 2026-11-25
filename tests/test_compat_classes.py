"""
Tests of util/compat_classes.py: the all-genes-at-once class counts
(get_all_gene_class_counts, used for class_counts.parquet) must agree with the
gene-by-gene path the models use (get_gene_class_counts).

The synthetic EMASE data cover the delicate cases: read classes shared by several
transcripts of a gene, a class compatible with different haplotypes through different
transcripts, classes shared with another gene (multimapping), a gene with no
multimapping at all, and transcripts with no reads before, between and after others.
"""

import numpy as np
import polars as pl
import pytest
import scipy.sparse

from util.compressed_emase import Emase
from util.compat_classes import get_all_gene_class_counts, get_gene_class_counts

HAPLOTYPES = list("ABCDEFGH")
ALL = set(HAPLOTYPES)

# Transcripts (rows of the EMASE matrices): t0, t5 and t6 get no reads
TRANSCRIPTS = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"]
TX_ANNOT = pl.DataFrame(
    {
        "transcript_id": TRANSCRIPTS,
        "gene_id": ["g0", "g1", "g1", "g2", "g3", "g2", "g4"],
    }
)
# Read classes: (transcript -> haplotypes it is compatible with)
CLASSES = [
    {"t1": ALL, "t2": ALL},  # both transcripts of g1 (isoform overlap)
    {"t1": {"A", "B"}},  # g1 via t1 only
    {"t1": {"A"}, "t2": {"B"}},  # g1: t1 in A, t2 in B -> gene-level {A, B}
    {"t1": ALL, "t4": ALL},  # g1 and g3 (multimapper)
    {"t3": {"C"}},  # g2 only; g2 has no multimapping reads at all
    {"t1": {"C"}, "t2": {"C"}, "t4": {"C", "D"}},  # two g1 transcripts + g3
]
COUNTS = {"DO002": [1, 0, 6, 2, 9, 1], "DO001": [10, 5, 3, 7, 4, 2]}


def make_emase(counts) -> Emase:
    n_loci, n_classes = len(TRANSCRIPTS), len(CLASSES)
    haps = {}
    for hi, hap in enumerate(HAPLOTYPES):
        m = np.zeros((n_loci, n_classes))
        for c, compat in enumerate(CLASSES):
            for tx, haps_compat in compat.items():
                if hap in haps_compat:
                    m[TRANSCRIPTS.index(tx), c] = 1
        haps[f"h{hi}"] = scipy.sparse.csr_matrix(m)
    return Emase(
        shape=(n_loci, len(HAPLOTYPES), n_classes),
        count=np.asarray(counts, dtype=float),
        haps=haps,
        lname=np.array([t.encode() for t in TRANSCRIPTS]),
        hname=HAPLOTYPES,
    )


@pytest.fixture(scope="module")
def emases():
    # deliberately not in sorted mouse order
    return {mouse: make_emase(counts) for mouse, counts in COUNTS.items()}


@pytest.fixture(scope="module")
def all_genes(emases):
    return get_all_gene_class_counts(TX_ANNOT, emases)


def as_table(compat_rows, counts_by_class, mouse_ids):
    """{gene-level class as haplotype letters: {mouse: count}}"""
    out = {}
    for compat, counts in zip(compat_rows, counts_by_class):
        key = "".join(h for h, c in zip(HAPLOTYPES, compat) if c)
        for mouse, count in zip(mouse_ids, counts):
            out.setdefault(key, {}).setdefault(mouse, 0)
            out[key][mouse] += count
    return out


def gene_row(all_genes, gene_id):
    rows = all_genes.filter(gene_id=gene_id).to_dicts()
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("gene_id", ["g1", "g2", "g3"])
def test_matches_gene_by_gene(emases, all_genes, gene_id):
    expected = get_gene_class_counts(gene_id, TX_ANNOT, emases)
    row = gene_row(all_genes, gene_id)
    assert as_table(row["compat"], row["count"], row["mouse_ids"]) == as_table(
        expected.compat, expected.counts.T, expected.ids
    )


def test_layout(all_genes):
    row = gene_row(all_genes, "g1")
    assert row["mouse_ids"] == sorted(COUNTS)
    n_classes = len(row["count"])
    assert all(len(c) == len(COUNTS) for c in row["count"])  # n_classes x n_samples
    assert all(len(c) == len(HAPLOTYPES) for c in row["compat"])  # n_classes x n_haps
    assert len(row["multimapping_counts"]) == n_classes


def test_shared_classes_counted_once(all_genes):
    """A class hitting several transcripts of a gene counts once for the gene."""
    row = gene_row(all_genes, "g1")
    totals = dict(zip(row["mouse_ids"], np.sum(row["count"], axis=0)))
    for mouse, counts in COUNTS.items():
        # g1 has classes 0, 1, 2, 3 and 5
        assert totals[mouse] == sum(counts[c] for c in [0, 1, 2, 3, 5])


def test_gene_without_multimapping_kept(all_genes):
    row = gene_row(all_genes, "g2")
    assert row["mouse_ids"] == sorted(COUNTS)
    table = as_table(row["compat"], row["count"], row["mouse_ids"])
    assert table == {"C": {m: c[4] for m, c in COUNTS.items()}}
    assert row["multimapping_counts"] == [[]]


def test_multimapping_counts(all_genes):
    row = gene_row(all_genes, "g1")
    mice = row["mouse_ids"]
    by_class = {}
    for compat, mm in zip(row["compat"], row["multimapping_counts"]):
        key = "".join(h for h, c in zip(HAPLOTYPES, compat) if c)
        by_class[key] = {e["multimapping_gene_id"]: e["counts"] for e in mm}
    # reads of g1's classes that are also compatible with g3: class 3 (all haps) and
    # class 5 (C), each counted once even though class 5 hits two g1 transcripts
    assert by_class["ABCDEFGH"] == {"g3": [COUNTS[m][3] for m in mice]}
    assert by_class["C"] == {"g3": [COUNTS[m][5] for m in mice]}
    assert by_class["AB"] == {"g3": [0 for _ in mice]}


def test_mismatched_haplotypes_rejected(emases):
    other = make_emase(COUNTS["DO001"])
    other.hname = list("HGFEDCBA")
    with pytest.raises(ValueError, match="haplotypes"):
        get_all_gene_class_counts(TX_ANNOT, {**emases, "DO003": other})
