# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD

import numpy as np


def reorder_mat(mat):
    """
    This function takes a distance matrix and reorders it such that
    the most similar entries (lowest distance) are found along the diagonal.

    If two rows have the same column as most similar, the one with
    the highest similarity uses that column, and the other row must
    use the next as-to-yet-unused most similar column.

    Output should be a reordered distance matrix, scaled so that the
    most similar entry gets a value of 1.0.
    """
    # Find the most similar column,.
    most_similar = mat.min(axis=1)
    # Distance may be zero when compared to self; if so,
    # use half of the second-closest entry. If all zeros, use
    # some other small value. Why these values? Emmm... :)
    most_similar[most_similar == 0] = np.minimum(  # avoid div by zero
        mat[mat > 0].min() / 2, np.sqrt(mat.std()))  # some small default num

    # Normalize the matrix.
    norm_mat = (mat.T / most_similar).T

    # Order from most to most certain.
    most_similar_idx = mat.argmin(axis=1)
    # Choose most similar by largest total distance score first
    # smallest second (hmmm... should this be smallest total distance
    # first? that would seem to indicate the most confusable).
    priority = np.argsort(norm_mat.sum(axis=1))[::-1]
    reidx = -1 * np.ones(priority.shape, dtype=int)  # new row index
    for ci, pi in enumerate(priority):
        msi = most_similar_idx[pi]
        if msi in reidx:  # collision; find the next-best choice.
            msi = filter(lambda ii: ii not in reidx,
                         np.argsort(norm_mat[pi]))[0]
        reidx[priority[ci]] = msi
    return norm_mat.T[reidx].T
