import traceback
import h5py
import xarray as xr


class GeneError:
    def __init__(self, exc: BaseException):
        self.message = str(exc)
        self.error_type = type(exc).__name__
        self.traceback = traceback.format_exc()  # call inside the except block


def save_results(results: dict, path: str):
    """Write {gene_id: InferenceData | GeneError} to one file, one group per gene."""
    mode = "w"
    for gene_id, obj in results.items():
        if isinstance(obj, GeneError):
            xr.Dataset(
                attrs={
                    "error": obj.message,
                    "error_type": obj.error_type,
                    "traceback": obj.traceback,
                }
            ).to_netcdf(path, group=f"{gene_id}/error", mode=mode, engine="h5netcdf")
            mode = "a"
            continue
        for group in obj.groups:
            ds = obj[group]
            ds = ds.to_dataset() if hasattr(ds, "to_dataset") else ds
            ds.to_netcdf(path, group=f"{gene_id}{group}", mode=mode, engine="h5netcdf")
            mode = "a"


def load_result(path: str, gene_id: str, groups=None):
    """Returns (InferenceData, None) on success or (None, error_attrs) on failure."""
    tree = xr.open_datatree(path, group=gene_id, engine="h5netcdf")
    if "error" in tree.children:
        return None, dict(tree["error"].attrs)
    if groups is not None:
        tree = xr.DataTree.from_dict({g: tree[g].to_dataset() for g in groups})
    return tree.load(), None


def list_errors(path: str) -> dict:
    with h5py.File(path, "r") as f:
        failed = [g for g in f.keys() if "error" in f[g].keys()]
    return {g: load_result(path, g)[1]["error"] for g in failed}
