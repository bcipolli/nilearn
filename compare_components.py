# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsuchida
# License: BSD

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from nilearn.image import iter_img, index_img, new_img_like
from nilearn.plotting import plot_stat_map
from six import string_types


def load_if_needed(img):
    if isinstance(img, string_types):
        img = nib.load(img)
    return img


def flip_img_lr(img):
    img = load_if_needed(img)
    # This won't work for all image formats! But
    # does work for those that we're working with...
    assert isinstance(img, nib.nifti1.Nifti1Image)
    return new_img_like(img, data=img.get_data()[::-1])


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
            plot_stat_map(comp, axes=ax,
                          title='%s[%d]' % (labels[ii], cis[ii]))
    plt.show()


if __name__ == '__main__':
    import os.path as op

    nii_dir = op.join('ica_nii', '20')
    img1 = op.join(nii_dir, 'R_ica_components.nii.gz')
    img2 = flip_img_lr(op.join(nii_dir, 'L_ica_components.nii.gz'))
    compare_components(images=[img1, img2], labels=['R', 'L'])
