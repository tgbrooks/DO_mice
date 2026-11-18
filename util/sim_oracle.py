"""
True ("oracle") alpha and q for genes of the class-level simulation, for diagnosing
the ASE models by plugging in the truth for an estimated quantity.

The truth comes from scripts/simulate_classes.py's outputs: q and its class_names in
gene_truths.parquet, per sample x class alpha in sample_truths.parquet (in the same
class order). Both are mapped to a model run's classes by class name, since the class
order can differ between datasets.
"""

import numpy as np
import polars as pl

from ase_buffering.ase_model_2steps import GeneData

ORACLE_MODES = ("none", "alpha", "q", "both")


class SimOracle:
    def __init__(self, mode: str, sim_tissue: str):
        if mode not in ORACLE_MODES or mode == "none":
            raise ValueError(f"oracle mode must be one of {ORACLE_MODES[1:]}, got {mode!r}")
        self.mode = mode
        self.gene_truths = pl.read_parquet(
            f"processed/{sim_tissue}/gene_truths.parquet"
        ).select("gene_id", "q", "class_names")
        self.sample_truths = pl.read_parquet(
            f"processed/{sim_tissue}/sample_truths.parquet"
        ).select("gene_id", "mouse_id", "alpha")

    def apply(self, gene_id: str, gene_data: GeneData):
        """Overwrite gene_data.alpha (modes alpha/both) and return the true q (modes
        q/both, else None).

        gene_data.info is left as computed from the estimated alpha, so which mice are
        considered informative does not change with the oracle.
        """
        truth = self.gene_truths.filter(gene_id=gene_id)
        if truth.height != 1:
            raise ValueError(f"No gene truth for {gene_id}")
        true_names = truth["class_names"].to_list()[0]
        mismatch = set(gene_data.class_names) ^ set(true_names)
        if mismatch:
            raise ValueError(f"Class mismatch with simulated truth: {sorted(mismatch)}")
        perm = [true_names.index(n) for n in gene_data.class_names]
        if self.mode in ("alpha", "both"):
            samples = self.sample_truths.filter(gene_id=gene_id)
            true_alpha = {
                m: np.asarray(a)[perm]
                for m, a in zip(samples["mouse_id"], samples["alpha"].to_list())
            }
            gene_data.alpha = np.stack([true_alpha[m] for m in gene_data.ids])
        if self.mode in ("q", "both"):
            return np.asarray(truth["q"].to_list()[0])[:, perm]
        return None
