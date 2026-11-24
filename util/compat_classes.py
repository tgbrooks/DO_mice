from dataclasses import dataclass
import polars as pl
import scipy.sparse
import numpy as np
import util.summarize_count_types


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

    def subset_samples(self, samples: list[str]) -> "CompatClassesDf":
        matches = np.array(self.ids)[None, :] == np.array(samples)[:, None]
        matched = matches.any(axis=1)
        if any(~matched):
            raise ValueError(
                f"Some samples where not in ids: eg: {np.array(samples)[~matched][:10]}"
            )
        idxs = np.argmax(matches, axis=1)
        return CompatClassesDf(
            compat=self.compat,
            counts=self.counts[idxs, :],
            ids=list(np.array(self.ids)[idxs]),
            haplotypes=self.haplotypes,
        )


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


def get_all_gene_class_counts(tx_annot, all_compatibility_classes) -> pl.DataFrame:
    """
    Get class counts for all genes at once as a polars DataFrame with columns:
    gene_id, count (n_classes x n_samples), compat (n_haps x n_classes), mouse_ids (n_samples)
    """

    _haps = pl.Enum(list("ABCDEFGH"))
    mouse_ids = list(all_compatibility_classes.keys())
    _mouse_ids = pl.Enum(mouse_ids)
    mouse_ids_df = pl.DataFrame({"mouse_id": mouse_ids}).select(
        pl.col("mouse_id").cast(_mouse_ids)
    )
    _tx_ids = pl.Enum(sorted(tx_annot["transcript_id"].unique()))
    tx_to_gene = tx_annot.select(pl.col("transcript_id").cast(_tx_ids), "gene_id")
    data = []
    for mouse_id, emase in all_compatibility_classes.items():
        mouse_data = []
        for hap, compat in zip(emase.hname, emase.haps.values()):
            indptr_df = pl.DataFrame(
                {
                    "transcript_id": emase.lname,
                    "indptr_start": compat.indptr[:-1],
                    # "indptr_end": compat.indptr[1:],
                }
            )
            indices_df = pl.DataFrame(
                {
                    "indptr": np.arange(len(compat.indices)),
                    "class_num": compat.indices,
                }
            )
            df = indices_df.join_asof(
                indptr_df,
                left_on="indptr",
                right_on="indptr_start",
                strategy="backward",
            )
            df = df.select(
                pl.lit(hap).alias("haplotype").cast(_haps),
                "class_num",
                pl.col("transcript_id").cast(str).cast(_tx_ids),
                count=emase.count[df["class_num"]].astype("int32"),
            )
            mouse_data.append(df)
        mouse_df = (
            pl.concat(mouse_data)
            .group_by("transcript_id", "class_num", "count")
            .agg(hap_list="haplotype")
        )
        # combine transcripts and classes that are the same for this gene
        mouse_gene_df = (
            mouse_df.join(tx_to_gene, "transcript_id")
            .group_by(
                "gene_id",
                "hap_list",
            )
            .agg(
                pl.col("count").sum(),
            )
        )
        data.append(
            mouse_gene_df.with_columns(
                pl.lit(mouse_id).alias("mouse_id").cast(_mouse_ids)
            )
        )
    gene_df = pl.concat(data)

    temp = []
    i = 0
    for (gene_id,), data in gene_df.group_by("gene_id"):
        if i % 1000 == 0:
            print(f"{i}: {gene_id}", flush=True)
        i += 1
        unique_classes = data.select("hap_list").unique()
        grid = unique_classes.join(
            mouse_ids_df,
            how="cross",
            maintain_order="right",
        )
        _df = (
            grid.join(
                data.select("mouse_id", "hap_list", "count"),
                ["mouse_id", "hap_list"],
                how="left",
                maintain_order="left",
            )
            .with_columns(pl.col("count").fill_null(0))
            .group_by("hap_list")
            .agg(
                pl.col("count"),
                mouse_ids=pl.col("mouse_id"),
            )
        )
        # Check consistency
        for _ids in _df["mouse_ids"]:
            assert (_ids == mouse_ids_df["mouse_id"]).all()
        temp.append(
            _df.select(
                pl.lit(gene_id).alias("gene_id"),
                pl.col("mouse_ids").first(),
                pl.col("count").implode(),
                compat=pl.concat_list(
                    [pl.lit(h).is_in(pl.col("hap_list")) for h in "ABCDEFGH"]
                ).implode(),
            )
        )
    by_gene = pl.concat(temp)

    return by_gene
