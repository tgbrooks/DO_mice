from dataclasses import dataclass
import polars as pl
import scipy.sparse
import numpy as np
from util.compressed_emase import Emase


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


def get_all_gene_class_counts(
    tx_annot: pl.DataFrame, all_compatibility_classes: dict[str, Emase]
) -> pl.DataFrame:
    """
    Get class counts for all genes at once, the same as get_gene_class_counts gives
    gene by gene, as a polars DataFrame with one row per gene and columns:
        gene_id
        mouse_ids            (n_samples) mouse ids, sorted; the order of all per-sample lists
        count                (n_classes x n_samples) reads of each class in each sample
        unique_count         (n_classes x n_samples) those of the reads that are not
                             compatible with any other gene (in tx_annot)
        compat               (n_classes x n_haplotypes) compatibility of each class with
                             each haplotype, haplotypes in the order of the EMASE files
        multimapping_counts  (n_classes) per class, a list of {multimapping_gene_id,
                             counts (n_samples)}: reads of the class that are also
                             compatible with another gene

    A read class belongs to a gene if it is compatible with any of the gene's
    transcripts, and is compatible with a haplotype of the gene if it is with any of the
    gene's transcripts in that haplotype. Each class is counted once per gene, however
    many of the gene's transcripts it hits. Classes with the same gene-level
    compatibility are combined, and every sample gets every class of the gene (0 where
    it has no reads).

    params:
        tx_annot: polars DataFrame with gene_id and transcript_id columns
        all_compatibility_classes: dictionary mapping mouse ids to Emase
    """
    haplotypes = list(next(iter(all_compatibility_classes.values())).hname)
    for mouse_id, emase in all_compatibility_classes.items():
        if list(emase.hname) != haplotypes:
            raise ValueError(
                f"{mouse_id} has haplotypes {emase.hname}, expected {haplotypes}"
            )
    _haps = pl.Enum(haplotypes)
    mouse_ids = list(sorted(all_compatibility_classes.keys()))
    _mouse_ids = pl.Enum(mouse_ids)
    mouse_ids_df = pl.DataFrame({"mouse_id": mouse_ids}).select(
        pl.col("mouse_id").cast(_mouse_ids)
    )
    _tx_ids = pl.Enum(sorted(tx_annot["transcript_id"].unique()))
    tx_to_gene = tx_annot.select(pl.col("transcript_id").cast(_tx_ids), "gene_id")
    data = []
    for mouse_id, emase in all_compatibility_classes.items():
        mouse_data = []
        transcript_names = emase.lname.astype(str)
        for hap, compat in zip(emase.hname, emase.haps.values()):
            # compat is CSR (transcripts x classes): row i's classes are
            # indices[indptr[i]:indptr[i + 1]], so this gives each entry's transcript
            row = np.repeat(np.arange(len(transcript_names)), np.diff(compat.indptr))
            mouse_data.append(
                pl.DataFrame(
                    {
                        "haplotype": pl.Series([hap] * len(row), dtype=_haps),
                        "class_num": compat.indices,
                        "transcript_id": pl.Series(transcript_names[row]).cast(_tx_ids),
                    }
                )
            )
        # Gene-level classes: compatible with a haplotype if any of the gene's
        # transcripts is (as extract_gene); one row per (gene, class), so each class is
        # counted once per gene however many of its transcripts it hits
        gene_classes = (
            pl.concat(mouse_data)
            .join(tx_to_gene, "transcript_id")
            .group_by("gene_id", "class_num")
            .agg(hap_list=pl.col("haplotype").unique().sort())
        )
        gene_classes = gene_classes.with_columns(
            count=emase.count[gene_classes["class_num"].to_numpy()].astype("int32")
        ).with_columns(
            # reads of classes that no other gene shares
            unique_count=pl.when(pl.len().over("class_num") == 1)
            .then(pl.col("count"))
            .otherwise(0)
            .cast(pl.Int32)
        )
        # combine classes with the same gene-level compatibility
        mouse_gene_df = gene_classes.group_by("gene_id", "hap_list").agg(
            pl.col("count").sum(),
            pl.col("unique_count").sum(),
        )
        # Record gene-level multimapping: reads of a gene's class (hap_list) that are
        # also compatible with another gene
        multimap = (
            gene_classes.join_where(
                gene_classes,
                pl.col("gene_id") != pl.col("gene_id_right"),
                pl.col("class_num") == pl.col("class_num_right"),
            )
            .group_by("gene_id", "gene_id_right", "hap_list")
            .agg(pl.col("count").sum())
        )
        mg = mouse_gene_df.join(
            multimap.group_by("gene_id", "hap_list").agg(
                pl.col("gene_id_right").alias("multimapping_gene_ids"),
                pl.col("count").alias("multimapping_counts"),
            ),
            ["gene_id", "hap_list"],
            how="left",
        ).with_columns(
            pl.col("multimapping_gene_ids").fill_null([]),
            pl.col("multimapping_counts").fill_null([]),
        )
        data.append(
            mg.with_columns(pl.lit(mouse_id).alias("mouse_id").cast(_mouse_ids))
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
                data.select("mouse_id", "hap_list", "count", "unique_count"),
                ["mouse_id", "hap_list"],
                how="left",
                maintain_order="left",
            )
            .with_columns(pl.col("count", "unique_count").fill_null(0))
            .group_by("hap_list")
            .agg(
                pl.col("count"),
                pl.col("unique_count"),
                mouse_ids=pl.col("mouse_id"),
            )
        )
        mm_grid = grid.join(
            data.explode("multimapping_gene_ids", empty_as_null=False)
            .select("multimapping_gene_ids")
            .unique(),
            how="cross",
            maintain_order="left",
        )
        _multimap = (
            data.explode(
                "multimapping_gene_ids",
                "multimapping_counts",
                empty_as_null=False,
            )
            .join(
                mm_grid,
                ["mouse_id", "hap_list", "multimapping_gene_ids"],
                maintain_order="right",
                how="right",
            )
            .group_by(
                "hap_list", "mouse_id", "multimapping_gene_ids", maintain_order=True
            )
            .agg(count=pl.col("multimapping_counts").sum())
            .rename({"multimapping_gene_ids": "multimapping_gene_id"})
            .group_by(
                "hap_list",
                "multimapping_gene_id",
            )
            .agg(
                counts=pl.col("count"),
                mouse_ids=pl.col("mouse_id"),
            )
        )
        # Check consistency: every per-sample list is in mouse_ids order
        for _ids in [*_df["mouse_ids"], *_multimap["mouse_ids"]]:
            assert (_ids == mouse_ids_df["mouse_id"]).all()
        _multimap = _multimap.group_by("hap_list").agg(
            multimapping_counts=pl.struct(
                pl.col("multimapping_gene_id"),
                pl.col("counts"),
            ).implode()
        )
        # left join: genes with no multimapping reads have no _multimap rows
        _df2 = _df.join(_multimap, "hap_list", how="left").with_columns(
            pl.col("multimapping_counts").fill_null([])
        )
        temp.append(
            _df2.select(
                pl.lit(gene_id).alias("gene_id"),
                pl.col("mouse_ids").first(),
                pl.col("count").implode(),
                pl.col("unique_count").implode(),
                compat=pl.concat_list(
                    [pl.lit(h).is_in(pl.col("hap_list")) for h in haplotypes]
                ).implode(),
                multimapping_counts=pl.col("multimapping_counts").implode(),
            )
        )
    by_gene = pl.concat(temp)

    return by_gene
