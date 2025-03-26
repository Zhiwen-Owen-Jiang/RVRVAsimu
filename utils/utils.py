import gzip
import bz2
import h5py
import logging
from functools import reduce
import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, cho_factor


def GetLogger(logpath):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(logpath, mode="w")
    # fh.setLevel(logging.INFO)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    log.addHandler(sh)

    return log


def sec_to_str(t):
    """Convert seconds to days:hours:minutes:seconds"""
    [d, h, m, s, n] = reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24]
    )
    f = ""
    if d > 0:
        f += "{D}d:".format(D=d)
    if h > 0:
        f += "{H}h:".format(H=h)
    if m > 0:
        f += "{M}m:".format(M=m)

    f += "{S}s".format(S=s)
    return f


def check_compression(dir):
    """
    Checking which compression should use

    Parameters:
    ------------
    dir: diretory to the dataset

    Returns:
    ---------
    openfunc: function to open the file
    compression: type of compression

    """
    if dir.endswith("gz") or dir.endswith("bgz"):
        compression = "gzip"
        openfunc = gzip.open
    elif dir.endswith("bz2"):
        compression = "bz2"
        openfunc = bz2.BZ2File
    elif (
        dir.endswith("zip")
        or dir.endswith("tar")
        or dir.endswith("tar.gz")
        or dir.endswith("tar.bz2")
    ):
        raise ValueError(
            "files with suffix .zip, .tar, .tar.gz, .tar.bz2 are not supported"
        )
    else:
        openfunc = open
        compression = None

    return openfunc, compression


def find_loc(num_list, target):
    """
    Finding the target number from a sorted list of numbers by binary search

    Parameters:
    ------------
    num_list: a sorted list of numbers
    target: the target number

    Returns:
    ---------
    the exact index or -1

    """
    l = 0
    r = len(num_list) - 1
    while l <= r:
        mid = (l + r) // 2
        if num_list[mid] == target:
            return mid
        elif num_list[mid] > target:
            r = mid - 1
        else:
            l = mid + 1
    return r


def inv(A):
    """
    Computing inverse for a symmetric and positive-definite matrix

    """
    cho_factors = cho_factor(A)
    A_inv = cho_solve(cho_factors, np.eye(A.shape[0]))

    return A_inv


def get_common_idxs(*idx_list, single_id=False):
    """
    Getting common indices among a list of double indices for subjects.
    Each element in the list must be a pd.MultiIndex instance.

    Parameters:
    ------------
    idx_list: a list of pd.MultiIndex
    single_id: if return single id as a list

    Returns:
    ---------
    common_idxs: common indices in pd.MultiIndex or list

    """
    common_idxs = None
    for idx in idx_list:
        if idx is not None:
            if not isinstance(idx, pd.MultiIndex):
                raise TypeError("index must be a pd.MultiIndex instance")
            if common_idxs is None:
                common_idxs = idx.copy()
            else:
                common_idxs = common_idxs.intersection(idx)
    if common_idxs is None:
        raise ValueError("no valid index provided")
    if len(common_idxs) == 0:
        raise ValueError("no common index exists")
    if single_id:
        common_idxs = common_idxs.get_level_values("IID").tolist()

    return common_idxs


class PermDistribution:
    def __init__(self, perm_file):
        self.bins = [(2,2), (3,3), (4,4), (5,5), (6,7), (8,9),
                     (10,11), (12,14), (15,20), (21,30), (31,60), (61,100)]
        self.sig_stats = {bin: dict() for bin in self.bins}
        self.count = {bin: 0 for bin in self.bins} 
        self.max_p = {bin: dict() for bin in self.bins}
        self.breaks = [bin[0] for bin in self.bins]

        h5file = h5py.File(f"{perm_file}", "r")
        all_bins = list_datasets(h5file)
        for bin_str in all_bins:
            bin1, bin2, voxel = tuple([int(x) for x in bin_str.split("_")])
            bin = tuple([bin1, bin2])
            data = h5file[bin_str]
            count = data.attrs["count"]
            self.sig_stats[bin][voxel] = data[:]
            self.count[bin] = count
            self.max_p[bin][voxel] = len(self.sig_stats[bin][voxel]) / count
        h5file.close()


def list_datasets(hdf5_file):
    dataset_names = []
    
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            dataset_names.append(name)
    
    hdf5_file.visititems(visitor_func)
    return dataset_names