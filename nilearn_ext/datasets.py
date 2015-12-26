# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsucihda
# License: BSD

import numpy as np
from nilearn import datasets


def fetch_neurovault(max_images=np.inf, query_server=True, fetch_terms=True,
                     map_types=['F map', 'T map', 'Z map'], **kwargs):
    """Give meaningful defaults, extra computations."""

    # Download matching images
    ss_all = datasets.fetch_neurovault(
        max_images=max_images, query_server=query_server, map_types=map_types,
        fetch_terms=fetch_terms, **kwargs)
    images = ss_all['images']

    if not fetch_terms:
        term_scores = None
    else:
        term_scores = ss_all['terms']

        # Clean & report term scores
        terms = np.array(term_scores.keys())
        term_matrix = np.asarray(term_scores.values())
        term_matrix[term_matrix < 0] = 0
        total_scores = np.mean(term_matrix, axis=1)

        print("Top 10 neurosynth terms from downloaded images:")
        for term_idx in np.argsort(total_scores)[-10:][::-1]:
            print('\t%-25s: %.2f' % (terms[term_idx], total_scores[term_idx]))

    return images, term_scores
