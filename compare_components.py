# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsuchida
# License: BSD

import os.path as op

import numpy as np
from matplotlib import pyplot as plt
from nilearn.image import iter_img, index_img
from nilearn.plotting import plot_stat_map

from hemisphere_masker import (MniHemisphereMasker, flip_img_lr)


def compare_components(images, labels):
    assert len(images) == 2
    assert len(labels) == 2

    print("Loading images.")
    images = [load_if_needed(img) for img in images]
    data = [img.get_data() for img in images]
    del data

    n_components = [img.shape[3] for img in images]
    assert n_components[0] == n_components[1]
    n_components = n_components[0]  # values @ 0 and 1 are the same

    print("Scoring closest components (by L1 norm)")
    score_mat = np.zeros((n_components, n_components))
    for c1i, comp1 in enumerate(iter_img(images[0])):
        for c2i, comp2 in enumerate(iter_img(images[1])):
            l1norm = np.abs(comp1.get_data() - comp2.get_data()).sum()
            score_mat[c1i, c2i] = l1norm

    # Find cross-image mapping
    most_similar_idx = score_mat.argmin(axis=1)

    print("Plotting results.")
    for c1i in range(n_components):
        cis = [c1i, most_similar_idx[c1i]]
        fh = plt.figure(figsize=(16, 10))

        for ii in [0, 1]:  # image index
            ax = fh.add_subplot(2, 1, ii + 1)
            comp = index_img(images[ii], cis[ii])

            # Use the 4 terms weighted most as a title
            terms = np.asarray(images[ii].terms.keys())
            ica_terms = np.asarray(images[ii].terms.values()).T
            ic_terms = ica_terms[cis[ii]]
            important_terms = terms[np.argsort(ic_terms)[-4:]]
            title = '%s[%d]: %s' % (
                labels[ii], cis[ii], ', '.join(important_terms[::-1]))

            plot_stat_map(comp, axes=ax, title=title)
    plt.show()


def prep_image_for_comparison(key, img=None):

    if key.lower() == 'l':
        img = flip_img_lr(img)

    elif key.lower() in ('r', 'test'):
        pass

    elif key.lower() == 'both':
        img = MniHemisphereMasker(hemisphere='R').mask_as_img(img)

    else:
        raise ValueError("Unknown key: %s" % key)

    return img
if __name__ == '__main__':
    import os.path as op
    import sys
    from hemisphere_masker import flip_img_lr

    # Select two images to compare
    key1 = 'R' if len(sys.argv) < 2 else sys.argv[1]
    key2 = 'L' if len(sys.argv) < 3 else sys.argv[2]

    img1 = get_and_normalize_image(key1)[1]
    img2 = get_and_normalize_image(key2)[1]
    compare_components(images=[img1, img2], labels=[key1, key2])
