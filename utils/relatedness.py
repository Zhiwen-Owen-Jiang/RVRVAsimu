import h5py
import logging
import numpy as np
import pandas as pd
import utils.dataset as ds


class LOCOpreds:
    """
    Reading LOCO LDR predictions

    """

    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        self.preds = self.file["ldr_loco_preds"]
        ids = self.file["id"][:]
        self.ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])
        self.id_idxs = np.arange(len(self.ids))
        self.ldr_col = (0, self.preds.shape[0])
        self.logger = logging.getLogger(__name__)

    def close(self):
        self.file.close()

    def select_ldrs(self, ldr_col=None):
        """
        ldr_col: [start, end) of zero-based LDR index

        """
        if ldr_col is not None:
            if ldr_col[1] <= self.preds.shape[0]:
                self.ldr_col = ldr_col
                self.logger.info(
                    f"Keeping LDR{ldr_col[0]+1} to LDR{ldr_col[1]} LOCO predictions."
                )
            else:
                raise ValueError(
                    f"{ldr_col[1]} is greater than #LDRs in LOCO predictions"
                )

    def keep(self, keep_idvs):
        """
        Keep subjects
        this method will only be invoked after extracting common subjects

        Parameters:
        ------------
        keep_idvs: a list or pd.MultiIndex of subject ids

        Returns:
        ---------
        self.id_idxs: numeric indices of subjects

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
        self.id_idxs = ids_df["id"].values
        if len(self.id_idxs) == 0:
            raise ValueError("no subject remaining in LOCO predictions")

    def data_reader(self, chr):
        """
        Reading LDR predictions for a chromosome

        """
        loco_preds_chr = self.preds[
            self.ldr_col[0] : self.ldr_col[1], :, chr - 1
        ].T  # (n, r)
        return loco_preds_chr[self.id_idxs]