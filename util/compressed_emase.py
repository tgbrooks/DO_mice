import h5py
import pickle
import io
from dataclasses import dataclass
import numpy as np
import scipy.sparse


@dataclass
class Emase:
    """
    Data as stored in a compressed.h5 output from emase / gbrs
    Contains a sparse format of the read groups - transcript compatibility matrix
    along with counts of each read group
    """

    shape: tuple[int, int, int]  # loci / haplotypes / read groups
    count: np.ndarray  # How many reads were in the read group
    haps: dict[
        str, scipy.sparse._csc.csc_matrix
    ]  # Compatibility of the read group with each transcript of each allele
    lname: np.ndarray  # names of the transcripts
    hname: list[str]  # haplotype names


def load_compressed_emase(H5: str, haplotypes: list[str]) -> Emase:
    f = h5py.File(H5, "r")
    count = np.asarray(f["count"])
    shape = pickle.load(io.BytesIO(f["/"].attrs["shape"]))
    n_loci, n_hap, n_read_groups = shape
    assert n_hap == len(haplotypes)
    haps = {
        hap: scipy.sparse.csr_matrix(
            (np.ones(len(f[hap]["indices"])), f[hap]["indices"], f[hap]["indptr"]),
            shape=(n_loci, n_read_groups),
        )
        for hap in haplotypes
    }
    lname = np.asarray(f["lname"])
    hname = pickle.load(io.BytesIO(f.attrs["hname"]))
    f.close()
    return Emase(
        shape=shape,
        count=count,
        haps=haps,
        lname=lname,
        hname=hname,
    )


def write_compressed_emase(emase: Emase, outpath: str):
    out = h5py.File(outpath, "w")
    out.attrs["hname"] = pickle.dumps(emase.hname, protocol=0)
    out.attrs["mtype"] = b"csc_matrix"
    out.attrs["VERSION"] = b"1.0"
    out.attrs["TITLE"] = b"Sparse3DMatrix"
    out.attrs["CLASS"] = b"GROUP"
    out.attrs["PYTABLES_FORMAT_VERSION"] = b"2.1"
    out.attrs["incidence_only"] = 1
    out["count"] = emase.count
    out["count"].attrs["CLASS"] = b"CARRAY"
    out["count"].attrs["TITLE"] = b""
    out["count"].attrs["VERSION"] = b"1.1"
    out["/"].attrs["shape"] = pickle.dumps(emase.shape, protocol=0)
    for hap, data in emase.haps.items():
        out.create_group(hap)
        out[hap].attrs["CLASS"] = b"GROUP"
        out[hap].attrs["TITLE"] = b"Sparse matrix components for {hap.encode()}"
        out[hap].attrs["VERSION"] = b"1.0"
        out[hap]["indptr"] = data.indptr
        out[hap]["indptr"].attrs["CLASS"] = b"CARRAY"
        out[hap]["indptr"].attrs["TITLE"] = b""
        out[hap]["indptr"].attrs["VERSION"] = b"1.1"
        out[hap]["indices"] = data.indices
        out[hap]["indices"].attrs["CLASS"] = b"CARRAY"
        out[hap]["indices"].attrs["TITLE"] = b""
        out[hap]["indices"].attrs["VERSION"] = b"1.1"
    out["lname"] = emase.lname
    out["lname"].attrs["CLASS"] = b"CARRAY"
    out["lname"].attrs["TITLE"] = b"Locus Names"
    out["lname"].attrs["VERSION"] = b"1.1"
    out.close()
