# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD

import numpy as np
from scipy import stats

def reorder_mat(mat, normalize=True):
    """
    This function takes a distance matrix and reorders it such that
    the most similar entries (lowest distance) are found along the diagonal.

    If two rows have the same column as most similar, the one with
    the highest similarity uses that column, and the other row must
    use the next as-to-yet-unused most similar column.

    Output should be a reordered distance matrix, scaled so that the
    most similar entry gets a value of 1.0.

    The function also returns the new index for the reordered matrix
    for plotting.
    """

    # Reorder rows
    row_reidx = np.argsort([np.diff(np.sort(row))[0]/np.sort(row)[0]
                           for row in mat])[::-1]
    mat = mat[row_reidx]

    # Find the most similar column,.
    most_similar = mat.min(axis=1)
    most_similar_idx = mat.argmin(axis=1)

    # Normalize the matrix, for column reordering.
    # set minimum to zero
    norm_mat = (mat.T - most_similar)  # transpose to work on cols

    # Order from most to most certain.
    # Choose most similar by largest total distance score first
    # smallest second (hmmm... should this be smallest total distance
    # first? that would seem to indicate the most confusable).
    priority = np.arange(len(row_reidx))
    col_reidx = -1 * np.ones(priority.shape, dtype=int)  # new row index
    for ci, pi in enumerate(priority):
        msi = most_similar_idx[pi]
        if msi in col_reidx:  # collision; find the next-best choice.
            msi = filter(lambda ii: ii not in col_reidx,
                         np.argsort(norm_mat.T[pi]))[0]
        col_reidx[pi] = msi

    # Now reorder according to top-to-least match.
    # import pdb; pdb.set_trace()
    out_mat = (norm_mat if normalize else mat.T)[col_reidx].T
    return out_mat, col_reidx, row_reidx  # col=x, row=y, thus the ordering

# first reorder rows, from smallest sim to largest.


def get_ic_terms(terms, ic_idx, sign = 1, standardize = False):
    
    term_vals = np.asarray(terms.values()).T
    ic_term_vals = term_vals[ic_idx]
    terms = np.asarray(terms.keys())
    
    ic_term_vals = sign*ic_term_vals
        
    if standardize:
        ic_term_vals = stats.zscore(ic_term_vals)
        
    return terms, ic_term_vals
        
def get_n_terms(terms, ic_idx, n_terms=4, top_bottom = 'top', sign=1):
    
    # Get the top or bottom n terms and return the terms

    (terms, ic_term_vals) = get_ic_terms(terms, ic_idx, sign=sign)
        
    if top_bottom == 'top':
        out_terms = terms[np.argsort(ic_term_vals)[:-(n_terms+1):-1]]
        
    elif top_bottom == 'bottom':
        out_terms = terms[np.argsort(ic_term_vals)[:n_terms]]
        
    return out_terms
