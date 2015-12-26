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
from nilearn_ext.datasets import fetch_neurovault
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


def main(dataset, keys=('R', 'L'), n_components=20, max_images=np.inf,
         scoring='l1norm', query_server=True,
         force=False, img_dir=None, plot_dir=None):
    this_dir = op.join(dataset, '%s-%dics' % (scoring, n_components))
    img_dir = img_dir or op.join('ica_nii', this_dir)
    plot_dir = plot_dir or op.join('ica_imgs', this_dir)

    # Download
    if dataset == 'neurovault':
        images, term_scores = fetch_neurovault(
            max_images=max_images, query_server=query_server)
    elif dataset == 'abide':
        images = datasets.fetch_abide_pcp(n_subjects=max_images)
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
    import warnings
    from argparse import ArgumentParser

    # Look for image computation errors
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    # Arg parsing
    hemi_choices = ['R', 'L', 'both']
    parser = ArgumentParser(description="Really?")
    parser.add_argument('key1', nargs='?', default='R', choices=hemi_choices)
    parser.add_argument('key2', nargs='?', default='L', choices=hemi_choices)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--offline', action='store_true', default=False)
    parser.add_argument('--components', nargs='?', type=int, default=20,
                        dest='n_components')
    parser.add_argument('--dataset', nargs='?', default='neurovault',
                        choices=['neurovault'])
    parser.add_argument('--scoring', nargs='?', default='scoring',
                        choices=['l1norm', 'l2norm', 'correlation'])
    args = vars(parser.parse_args())

    # Alias args
    keys = args.pop('key1'), args.pop('key2')
    query_server = not args.pop('offline')
    main(keys=keys, query_server=query_server, **args)

    plt.show()
