"""Tests of util/paralogs.py: paralog pair selection and matched null pairs."""

import numpy as np
import polars as pl
import pytest

from util.paralogs import (
    families,
    null_pairs,
    ordered,
    read_mouse_paralogs,
    select_pairs,
    unordered_pairs,
)

HEADER = [
    "gene_stable_id", "protein_stable_id", "species", "identity", "homology_type",
    "homology_gene_stable_id", "homology_protein_stable_id", "homology_species",
    "homology_identity", "dn", "ds", "goc_score", "wga_coverage", "is_high_confidence",
    "homology_id",
]  # fmt: skip


def row(g1, g2, id1, id2, htype="other_paralog", sp2="mus_musculus"):
    return [g1, "P1", "mus_musculus", id1, htype, g2, "P2", sp2, id2] + ["NULL"] * 5 + ["1"]


ROWS = [
    row("gA", "gB", 60, 55),  # good pair, chr 1 / chr 2
    row("gB", "gA", 55, 60),  # same pair, other direction
    row("gA", "gB", 60, 55, htype="ortholog_one2one"),  # not a paralog row
    row("gA", "hsX", 90, 90, sp2="homo_sapiens"),  # not mouse-mouse
    row("gC", "gD", 50, 50, htype="gene_split"),  # gene split: excluded
    row("gE", "gF", 80, 30),  # low identity one way: excluded
    row("gG", "gH", 70, 70, htype="within_species_paralog"),  # same chromosome
    row("gI", "gX", 70, 70),  # gX on chromosome X
    row("gJ", "gK", 70, 70),  # gK not expressed
    # big family f0-f5 (6 genes), on different chromosomes: excluded with max size 5
    *[row(f"f{i}", f"f{j}", 70, 70) for i in range(6) for j in range(i + 1, 6)],
]
CHROM = {
    "gA": "1", "gB": "2", "gC": "3", "gD": "4", "gE": "5", "gF": "6", "gG": "7", "gH": "7",
    "gI": "8", "gX": "X", "gJ": "9", "gK": "10",
    **{f"f{i}": str(11 + i) for i in range(6)},
    **{f"n{i}": str(1 + i % 19) for i in range(60)},  # other expressed genes
}  # fmt: skip
CHROMOSOMES = [str(c) for c in range(1, 20)]


@pytest.fixture(scope="module")
def paralogs(tmp_path_factory):
    path = tmp_path_factory.mktemp("compara") / "homologies.tsv.gz"
    import gzip

    with gzip.open(path, "wt") as f:
        f.write("\t".join(HEADER) + "\n")
        for r in ROWS:
            f.write("\t".join(map(str, r)) + "\n")
    return read_mouse_paralogs(str(path))


@pytest.fixture(scope="module")
def gene_chrom():
    return pl.DataFrame({"gene_id": list(CHROM), "chrom": list(CHROM.values())})


@pytest.fixture(scope="module")
def expression():
    expr = {g: 100.0 + i for i, g in enumerate(CHROM)}
    expr["gK"] = 1.0
    return pl.DataFrame({"gene_id": list(expr), "expression": list(expr.values())})


def test_read_keeps_mouse_paralogs_only(paralogs):
    assert set(paralogs["homology_type"]) <= {"within_species_paralog", "other_paralog"}
    assert "hsX" not in paralogs["gene_2"].to_list()
    assert unordered_pairs(paralogs).filter(gene_1="gA", gene_2="gB").height == 1


def test_families():
    pairs = pl.DataFrame({"gene_1": ["a", "b", "x"], "gene_2": ["b", "c", "y"]})
    fam = {r["gene_id"]: (r["family_id"], r["family_size"]) for r in families(pairs).to_dicts()}
    assert fam["a"] == fam["b"] == fam["c"] == ("a", 3)
    assert fam["x"] == fam["y"] == ("x", 2)


def test_select_pairs(paralogs, gene_chrom, expression):
    pairs, report = select_pairs(
        paralogs,
        gene_chrom,
        expression,
        min_identity=40,
        max_family_size=5,
        min_expression=20,
        chromosomes=CHROMOSOMES,
    )
    # gE-gF: identity; gG-gH: same chromosome; gI-gX: chromosome X; gJ-gK: expression;
    # f*: family too big; gC-gD: gene split
    assert pairs.select("gene_1", "gene_2").rows() == [("gA", "gB")]
    assert [n for _, n in report] == sorted([n for _, n in report], reverse=True)
    both = ordered(pairs)
    assert set(both.select("gene_a", "gene_b").rows()) == {("gA", "gB"), ("gB", "gA")}


def test_null_pairs(paralogs, gene_chrom, expression):
    pairs, _ = select_pairs(
        paralogs, gene_chrom, expression, min_identity=40, max_family_size=5,
        min_expression=20, chromosomes=CHROMOSOMES,
    )  # fmt: skip
    candidates = (
        expression.filter(pl.col("expression") >= 20)
        .join(gene_chrom, "gene_id")
        .filter(pl.col("chrom").is_in(CHROMOSOMES))
    )
    all_paralogs = unordered_pairs(paralogs)
    nulls = null_pairs(
        ordered(pairs), candidates, all_paralogs, n_per_pair=5, window=20, seed=0
    )
    assert nulls.height == 10
    paralog_set = set(all_paralogs.select("gene_1", "gene_2").rows())
    for r in nulls.to_dicts():
        assert r["gene_a"] != r["gene_b"]
        assert CHROM[r["gene_a"]] != CHROM[r["gene_b"]]
        assert tuple(sorted([r["gene_a"], r["gene_b"]])) not in paralog_set
    assert nulls.select("gene_a", "gene_b").is_unique().all()
    # drawn near the matched gene in expression rank
    rank = {g: i for i, g in enumerate(candidates.sort("expression")["gene_id"])}
    for r in nulls.to_dicts():
        assert abs(rank[r["gene_a"]] - rank[r["matched_a"]]) < 20
        assert abs(rank[r["gene_b"]] - rank[r["matched_b"]]) < 20
    # reproducible
    again = null_pairs(ordered(pairs), candidates, all_paralogs, n_per_pair=5, window=20, seed=0)
    assert again.equals(nulls)
