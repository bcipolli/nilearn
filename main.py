# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsuchida
# License: BSD

import os.path as op

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.image import index_img, iter_img

from nibabel_ext import NiftiImageWithTerms
from nilearn_ext.datasets import fetch_neurovault_images_and_terms
from nilearn_ext.decomposition import compare_components, generate_components
from nilearn_ext.masking import join_bilateral_rois
from nilearn_ext.plotting import (plot_comparisons, plot_components,
                                  save_and_close)
from nilearn_ext.utils import reorder_mat


def load_or_generate_components(hemi, out_dir='.', plot_dir=None,
                                *args, **kwargs):
    # Only re-run if image doesn't exist.
    img_path = op.join(out_dir, '%s_ica_components.nii.gz' % hemi)
    if not kwargs.get('force') and op.exists(img_path):
        img = NiftiImageWithTerms.from_filename(img_path)

    else:
        img = generate_components(hemi=hemi, out_dir=out_dir, *args, **kwargs)
        plot_components(img, hemi=hemi, out_dir=plot_dir)
    return img


def mix_and_match_bilateral_components(**kwargs):
    # LR image: do ICA for L, then R, then match up & combine
    # into a set of bilateral images.
    R_img = load_or_generate_components(hemi='R', **kwargs)  # noqa
    L_img = load_or_generate_components(hemi='L', **kwargs)  # noqa

    # Match
    score_mat = compare_components(images=(R_img, L_img),
                                   labels=('R', 'L'))
    most_similar_idx = score_mat.argmin(axis=1)

    # Mix
    terms = R_img.terms.keys()
    term_scores = []
    bilat_imgs = []
    for rci, R_comp_img in enumerate(iter_img(R_img)):
        lci = most_similar_idx[rci]
        L_comp_img = index_img(L_img, lci)  # noqa
        # combine images
        bilat_imgs.append(join_bilateral_rois(R_comp_img, L_comp_img))
        # combine terms
        if terms:
            term_scores.append([(R_img.terms[t][rci] +
                                 L_img.terms[t][lci]) / 2
                                for t in terms])

    # Squash into single image
    img = nib.concat_images(bilat_imgs)
    if terms:
        img.terms = dict(zip(terms, np.asarray(term_scores).T))
    return img


def main(dataset, keys=('R', 'L'), n_components=20, n_images=np.inf,
         scoring='l1norm',
         force=False, img_dir=None, plot_dir=None):
    this_dir = op.join(dataset, '%s-%dics' % (scoring, n_components))
    img_dir = img_dir or op.join('ica_nii', this_dir)
    plot_dir = plot_dir or op.join('ica_imgs', this_dir)

    # Download
    if dataset == 'neurovault':
        images, term_scores = fetch_neurovault_images_and_terms(
            n_images=n_images, query_server=False)
    elif dataset == 'abide':
        images = datasets.fetch_abide_pcp(n_subjects=n_images)
        term_scores = None

    # Analyze images
    print("Running all analyses on both hemis together, and each separately.")
    imgs = dict()
    kwargs = dict(images=images, term_scores=term_scores,
                  n_components=n_components,
                  out_dir=img_dir, plot_dir=plot_dir)
    for key in keys:
        if key.lower() in ('rl', 'lr'):
            imgs[key] = mix_and_match_bilateral_components(**kwargs)
        else:
            imgs[key] = load_or_generate_components(hemi=key, **kwargs)

    # Show confusion matrix
    score_mat = compare_components(images=imgs.values(), labels=imgs.keys(),
                                   scoring=scoring)
    fh = plt.figure(figsize=(10, 10))
    fh.gca().matshow(reorder_mat(score_mat))
    save_and_close(out_path=op.join(plot_dir, '%s_%s_simmat.png' % keys))

    # Get the requested images
    plot_comparisons(images=imgs.values(), labels=imgs.keys(),
                     score_mat=score_mat, out_dir=plot_dir)


if __name__ == '__main__':
    import sys
    import warnings

    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    # Select two images to compare
    key1 = 'R' if len(sys.argv) < 2 else sys.argv[1]
    key2 = 'L' if len(sys.argv) < 3 else sys.argv[2]

    # Settings
    n_components = 20
    main(dataset='neurovault', keys=(key1, key2), n_components=n_components,
         scoring='correlation')
    plt.show()
