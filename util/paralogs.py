"""
Paralog pairs for the paralog compensation analysis (steps/paralogs.smk).

From Ensembl Compara mouse paralogs, keep pairs that are similar enough, not in a
large (mostly old) gene family, on different autosomes (no tandem duplicates, no LD
between the two loci) and both expressed. Each unordered pair gives two ordered pairs
(A -> B and B -> A): the effect of gene A's locus on gene B's expression.

Null pairs: random gene pairs, matched on the expression of A and of B, on different
autosomes and not paralogs of each other, analysed in exactly the same way.
"""

import numpy as np
import polars as pl

PARALOG_TYPES = ("within_species_paralog", "other_paralog")


def read_mouse_paralogs(path: str) -> pl.DataFrame:
    """Mouse-mouse paralog rows of an Ensembl Compara homologies TSV."""
    return (
        pl.read_csv(
            path,
            separator="\t",
            null_values="NULL",
            columns=[
                "gene_stable_id",
                "species",
                "identity",
                "homology_type",
                "homology_gene_stable_id",
                "homology_species",
                "homology_identity",
            ],
            schema_overrides={"identity": pl.Float64, "homology_identity": pl.Float64},
        )
        .filter(
            pl.col("species") == "mus_musculus",
            pl.col("homology_species") == "mus_musculus",
            pl.col("homology_type").is_in(PARALOG_TYPES),
        )
        .select(
            gene_1="gene_stable_id",
            gene_2="homology_gene_stable_id",
            identity_1="identity",
            identity_2="homology_identity",
            homology_type="homology_type",
        )
    )


def unordered_pairs(paralogs: pl.DataFrame) -> pl.DataFrame:
    """One row per unordered gene pair (gene_1 < gene_2), with the lower identity."""
    return (
        paralogs.filter(pl.col("gene_1") != pl.col("gene_2"))
        .select(
            gene_1=pl.min_horizontal("gene_1", "gene_2"),
            gene_2=pl.max_horizontal("gene_1", "gene_2"),
            identity=pl.min_horizontal("identity_1", "identity_2"),
            homology_type="homology_type",
        )
        .group_by("gene_1", "gene_2")
        .agg(pl.col("identity").min(), pl.col("homology_type").first())
        .sort("gene_1", "gene_2")
    )


def families(pairs: pl.DataFrame) -> pl.DataFrame:
    """Connected components of the pair graph: gene_id, family_id, family_size."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs.select("gene_1", "gene_2").iter_rows():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    genes = sorted(parent)
    df = pl.DataFrame({"gene_id": genes, "family_id": [find(g) for g in genes]})
    return df.with_columns(family_size=pl.len().over("family_id"))


def select_pairs(
    paralogs: pl.DataFrame,
    gene_chrom: pl.DataFrame,
    expression: pl.DataFrame,
    *,
    min_identity: float,
    max_family_size: int,
    min_expression: float,
    chromosomes: list[str],
) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
    """
    Unordered paralog pairs passing all filters, and the number of pairs left after
    each filter (for reporting).

    gene_chrom: gene_id, chrom. expression: gene_id, expression (e.g. median reads).
    """
    report = []
    pairs = unordered_pairs(paralogs)
    report.append(("mouse paralog pairs", pairs.height))
    pairs = pairs.filter(pl.col("identity") >= min_identity)
    report.append((f"identity >= {min_identity}% (both directions)", pairs.height))
    fam = families(pairs)
    pairs = (
        pairs.join(fam.select(gene_1="gene_id", family_id="family_id", family_size="family_size"), "gene_1")
        .filter(pl.col("family_size") <= max_family_size)
    )
    report.append((f"family size <= {max_family_size}", pairs.height))
    chrom = gene_chrom.select("gene_id", pl.col("chrom").cast(str))
    pairs = (
        pairs.join(chrom.rename({"gene_id": "gene_1", "chrom": "chrom_1"}), "gene_1")
        .join(chrom.rename({"gene_id": "gene_2", "chrom": "chrom_2"}), "gene_2")
        .filter(
            pl.col("chrom_1").is_in(chromosomes),
            pl.col("chrom_2").is_in(chromosomes),
        )
    )
    report.append((f"both in the reference, on chromosomes {','.join(chromosomes)}", pairs.height))
    pairs = pairs.filter(pl.col("chrom_1") != pl.col("chrom_2"))
    report.append(("on different chromosomes", pairs.height))
    expr = expression.select("gene_id", "expression")
    pairs = (
        pairs.join(expr.rename({"gene_id": "gene_1", "expression": "expression_1"}), "gene_1")
        .join(expr.rename({"gene_id": "gene_2", "expression": "expression_2"}), "gene_2")
        .filter(
            pl.col("expression_1") >= min_expression,
            pl.col("expression_2") >= min_expression,
        )
    )
    report.append((f"both expressed (>= {min_expression})", pairs.height))
    return pairs.sort("gene_1", "gene_2"), report


def ordered(pairs: pl.DataFrame) -> pl.DataFrame:
    """Both directions of each unordered pair, as gene_a (locus) -> gene_b (response)."""
    keep = ["identity", "homology_type", "family_id", "family_size"]
    return pl.concat(
        [
            pairs.select(gene_a="gene_1", gene_b="gene_2", *keep),
            pairs.select(gene_a="gene_2", gene_b="gene_1", *keep),
        ]
    ).sort("gene_a", "gene_b")


def null_pairs(
    paralog_ordered: pl.DataFrame,
    candidates: pl.DataFrame,
    all_paralogs: pl.DataFrame,
    *,
    n_per_pair: int,
    window: int,
    seed: int,
    max_tries: int = 1000,
) -> pl.DataFrame:
    """
    For each ordered paralog pair (A, B), n_per_pair random pairs (A', B') with A' and
    B' drawn from the `window` candidate genes nearest in expression rank to A and to B,
    on different chromosomes, distinct, and not paralogs of each other (any identity).
    Each null pair is drawn at most once.

    candidates: gene_id, chrom, expression (expressed autosomal genes).
    all_paralogs: unordered pairs (gene_1, gene_2) of all mouse paralogs.
    """
    rng = np.random.default_rng(seed)
    cand = candidates.sort("expression", "gene_id")
    genes = cand["gene_id"].to_list()
    chroms = cand["chrom"].cast(str).to_list()
    rank = {g: i for i, g in enumerate(genes)}
    paralog_set = {(a, b) for a, b in all_paralogs.select("gene_1", "gene_2").iter_rows()}
    paralog_set |= {(b, a) for a, b in paralog_set}
    n = len(genes)

    def near(gene):
        r = rank[gene]
        lo = max(0, min(r - window // 2, n - window))
        return np.arange(lo, min(n, lo + window))

    rows, seen = [], set()
    for a, b in paralog_ordered.select("gene_a", "gene_b").iter_rows():
        if a not in rank or b not in rank:
            continue
        near_a, near_b = near(a), near(b)
        for k in range(n_per_pair):
            for _ in range(max_tries):
                i, j = rng.choice(near_a), rng.choice(near_b)
                ga, gb = genes[i], genes[j]
                if (
                    ga != gb
                    and chroms[i] != chroms[j]
                    and (ga, gb) not in paralog_set
                    and (ga, gb) not in seen
                ):
                    seen.add((ga, gb))
                    rows.append({"gene_a": ga, "gene_b": gb, "matched_a": a, "matched_b": b})
                    break
    schema = {"gene_a": pl.String, "gene_b": pl.String, "matched_a": pl.String, "matched_b": pl.String}
    return pl.DataFrame(rows, schema=schema)
