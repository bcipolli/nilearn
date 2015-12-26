# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD

import numpy as np


def reorder_mat(mat):
    # Try to put items on the diagonal.
    most_similar_idx = mat.argmin(axis=1)
    most_similar = mat.min(axis=1)
    norm_mat = (mat.T / most_similar).T

    # Order from most to most certain.
    priority = np.argsort(norm_mat.sum(axis=1))[::-1]
    reidx = -1 * np.ones(priority.shape, dtype=int)
    for ci, pi in enumerate(priority):
        msi = most_similar_idx[pi]
        if msi in reidx:  # overlap; find the next-best choice.
            msi = filter(lambda ii: ii not in reidx,
                         np.argsort(norm_mat[pi]))[0]
        reidx[priority[ci]] = msi
    return norm_mat.T[reidx].T
