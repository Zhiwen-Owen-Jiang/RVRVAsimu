import h5py
import logging
import numpy as np
import pandas as pd
import utils.dataset as ds


class NullModel:
    """
    Reading and processing null model

    """

    def __init__(self, file_path):
        with h5py.File(file_path, "r") as file:
            self.covar = file["covar"][:]
            self.resid_ldr = file["resid_ldr"][:]
            self.bases = file["bases"][:]
            ids = file["id"][:]

        self.n_voxels, self.n_ldrs = self.bases.shape
        self.n_subs = self.covar.shape[0]
        self.ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])
        self.id_idxs = np.arange(self.n_subs)
        self.voxel_idxs = np.arange(self.n_voxels)
        self.logger = logging.getLogger(__name__)

    def select_ldrs(self, n_ldrs=None):
        if n_ldrs is not None:
            if n_ldrs <= self.n_ldrs:
                self.n_ldrs = n_ldrs
                self.resid_ldr = self.resid_ldr[:, :n_ldrs]
                self.bases = self.bases[:, :n_ldrs]
                self.logger.info(f"Keeping the top {n_ldrs} LDRs and bases.")
            else:
                raise ValueError("--n-ldrs is greater than #LDRs in null model")

    def select_voxels(self, voxel_idxs=None):
        if voxel_idxs is not None:
            if np.max(voxel_idxs) < self.n_voxels:
                self.voxel_idxs = voxel_idxs
                self.bases = self.bases[voxel_idxs]
                self.logger.info(f"{len(voxel_idxs)} voxel(s) included.")
            else:
                raise ValueError("--voxels index (one-based) out of range")

    def keep(self, keep_idvs):
        """
        Keep subjects
        this method will only be invoked after extracting common subjects

        Parameters:
        ------------
        keep_idvs: a list or pd.MultiIndex of subject ids to keep

        """
        if isinstance(keep_idvs, list):
            keep_idvs = pd.MultiIndex.from_arrays(
                [keep_idvs, keep_idvs], names=["FID", "IID"]
            )
        common_ids = ds.get_common_idxs(keep_idvs, self.ids).get_level_values("IID")
        ids_df = pd.DataFrame(
            {"id": self.id_idxs}, index=self.ids.get_level_values("IID")
        )
        ids_df = ids_df.loc[common_ids]
        id_idxs = ids_df["id"].values
        self.resid_ldr = self.resid_ldr[id_idxs]
        self.covar = self.covar[id_idxs]
        if self.resid_ldr.shape[0] == 0 or self.covar.shape[0] == 0:
            raise ValueError("no subject remaining in the null model")

    def remove_dependent_columns(self):
        """
        Removing dependent columns from covariate matrix

        """
        rank = np.linalg.matrix_rank(self.covar)
        if rank < self.covar.shape[1]:
            _, R = np.linalg.qr(self.covar)
            independent_columns = np.where(np.abs(np.diag(R)) > 1e-10)[0]
            self.covar = self.covar[:, independent_columns]
            if self.covar.shape[1] == 0:
                raise ValueError("no covariate remaining in the null model")