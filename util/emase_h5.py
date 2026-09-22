import h5py
import pickle
import io
from dataclasses import dataclass
import numpy as np
import scipy.sparse


@dataclass
class ReadsEmase:
    """
    Data as stored in a (merged but not compressed) .h5 output from emase
    Contains a sparse format of the reads - transcript compatibility matrix
    """

    shape: tuple[int, int, int]  # loci / haplotypes / reads
    haps: dict[
        str, scipy.sparse._csc.csc_matrix
    ]  # Compatibility of the read group with each transcript of each allele
    lname: np.ndarray  # names of the transcripts
    rname: np.ndarray  # names of the reads
    hname: list[str]  # haplotype names


def load_reads_emase(H5: str) -> ReadsEmase:
    f = h5py.File(H5, "r")
    shape = pickle.load(io.BytesIO(f["/"].attrs["shape"]))
    n_loci, n_hap, n_read_groups = shape
    haplotypes = [f"h{i}" for i in range(n_hap)]
    haps = {
        hap: scipy.sparse.csr_array(
            (np.ones(len(f[hap]["indices"])), f[hap]["indices"], f[hap]["indptr"]),
            shape=(n_loci, n_read_groups),
        )
        for hap in haplotypes
    }
    lname = np.asarray(f["lname"])
    rname = np.asarray(f["rname"])
    hname = pickle.load(io.BytesIO(f.attrs["hname"]))
    assert n_hap == len(hname)
    f.close()
    return ReadsEmase(
        shape=shape,
        haps=haps,
        lname=lname,
        rname=rname,
        hname=hname,
    )
