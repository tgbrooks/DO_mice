"""
True ("oracle") alpha and q for genes of the class-level simulation, for diagnosing
the ASE models by plugging in the truth for an estimated quantity.

Truth as scripts/simulate_classes.py defines it:
  q     = q_hat of the real reference-tissue data (min_genotype_confidence=0.9)
  alpha = per sample x class values saved in sample_truths.parquet
Both are in the reference data's class order and are mapped to the simulated data's
classes by class name (the simulated classes come from the same compat matrix).
"""

import numpy as np
import polars as pl

from util.compressed_emase import load_compressed_emase
from util.compat_classes import get_gene_class_counts
from ase_buffering.ase_model_2steps import GeneData, prepare_gene_data

ORACLE_MODES = ("none", "alpha", "q", "both")


class SimOracle:
    def __init__(
        self,
        mode: str,
        sim_tissue: str,
        ref_tissue: str,
        tx_annot: pl.DataFrame,
        genotypes: pl.DataFrame,
        n_haplotypes: int,
        ref_outlier_ids: list[str],
    ):
        if mode not in ORACLE_MODES or mode == "none":
            raise ValueError(f"oracle mode must be one of {ORACLE_MODES[1:]}, got {mode!r}")
        self.mode = mode
        self.tx_annot = tx_annot
        self.genotypes = genotypes
        self.sample_truths = pl.read_parquet(
            f"processed/{sim_tissue}/sample_truths.parquet"
        )
        self.ref_effective_lengths = pl.read_parquet(
            f"processed/{ref_tissue}/effective_lengths.by_gene.parquet"
        )
        # Needed for both modes: true q, and the class order true alpha is stored in
        ref_mice = sorted(
            pl.read_parquet(f"processed/{ref_tissue}/allele_unique_reads.parquet")[
                "mouse_id"
            ].unique()
        )
        self.ref_compatibility_classes = {
            mouse_id: load_compressed_emase(
                f"processed/{ref_tissue}/gbrs/{mouse_id}.compressed.h5",
                [f"h{i}" for i in range(n_haplotypes)],
            )
            for mouse_id in ref_mice
            if mouse_id not in ref_outlier_ids
        }

    def _reference(self, gene_id):
        """Reference-tissue q_hat and class names, as simulate_classes.py builds them."""
        gcc = get_gene_class_counts(gene_id, self.tx_annot, self.ref_compatibility_classes)
        ref = prepare_gene_data(
            gene_id,
            gcc,
            self.genotypes,
            self.ref_effective_lengths.filter(gene_id=gene_id).select(
                "mouse_id", "haplotype", "effective_length"
            ),
            config=dict(min_genotype_confidence=0.9, drop_all_zeros=True),
        )
        return ref.q_hat, ref.class_names

    def apply(self, gene_id: str, gene_data: GeneData):
        """Overwrite gene_data.alpha (modes alpha/both) and return the true q (modes
        q/both, else None).

        gene_data.info is left as computed from the estimated alpha, so which mice are
        considered informative does not change with the oracle.
        """
        q_true, ref_names = self._reference(gene_id)
        mismatch = set(gene_data.class_names) ^ set(ref_names)
        if mismatch:
            raise ValueError(f"Class mismatch with reference data: {sorted(mismatch)}")
        perm = [ref_names.index(n) for n in gene_data.class_names]
        if self.mode in ("alpha", "both"):
            truth = self.sample_truths.filter(gene_id=gene_id)
            if truth.height == 0:
                raise ValueError(f"No sample truths for {gene_id}")
            true_alpha = {
                m: np.asarray(a)[perm]
                for m, a in zip(truth["mouse_id"], truth["alpha"].to_list())
            }
            gene_data.alpha = np.stack([true_alpha[m] for m in gene_data.ids])
        if self.mode in ("q", "both"):
            return q_true[:, perm]
        return None
