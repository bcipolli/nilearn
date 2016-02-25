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
from nilearn_ext.plotting import (plot_component_comparisons, plot_components,
                                  plot_comparison_matrix)


def load_or_generate_components(hemi, out_dir='.', plot_dir=None, force=False,
                                *args, **kwargs):
    """ Load an image and return if it exists, otherwise compute via ICA"""

    # Only re-run if image doesn't exist.
    img_path = op.join(out_dir, '%s_ica_components.nii.gz' % hemi)
    if not force and op.exists(img_path):
        img = NiftiImageWithTerms.from_filename(img_path)

    else:
        img = generate_components(hemi=hemi, out_dir=out_dir, *args, **kwargs)
        png_dir = op.join(out_dir, 'png')
        plot_components(img, hemi=hemi, out_dir=png_dir)
    return img


def mix_and_match_bilateral_components(**kwargs):
    """Run ICA on R,L; then match up components and
    and concatenate matched components into a full-brain picture.
    """

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


def get_dataset(dataset, max_images=np.inf, **kwargs):
    """Retrieve & normalize dataset from nilearn"""
    # Download
    if dataset == 'neurovault':
        images, term_scores = fetch_neurovault(max_images=max_images, **kwargs)

    elif dataset == 'abide':
        dataset = datasets.fetch_abide_pcp(
            n_subjects=min(94, max_images), **kwargs)
        images = [{'local_path': p} for p in dataset['func_preproc']]
        term_scores = None

    elif dataset == 'nyu':
        dataset = datasets.fetch_nyu_rest(
            n_subjects=min(25, max_images), **kwargs)
        images = [{'local_path': p} for p in dataset['func']]
        term_scores = None

    else:
        raise ValueError("Unknown dataset: %s" % dataset)
    return images, term_scores


def main(dataset, n_components=20, max_images=np.inf,
         scoring='l1norm', query_server=True,
         force=False, nii_dir=None, plot_dir=None, random_state=42):
    """Compute components, then run requested comparisons"""

    # Output directories
    nii_dir = nii_dir or op.join('ica_nii', dataset, str(n_components))
    plot_dir = plot_dir or op.join('ica_imgs', dataset,
                                   '%s-%dics' % (scoring, n_components))

    images, term_scores = get_dataset(dataset, max_images=max_images,
                                      query_server=query_server)

    # Components are generated for R-, L-only, and whole brain images.
    # R- and L- only components are then compared against wb.
    comparisons = [('wb','r'),('wb','l')]
    for comp in comparisons:
        
        # Load or generate components
        imgs = []
        kwargs = dict(images=[im['local_path'] for im in images],
                      n_components=n_components, term_scores=term_scores, 
                      out_dir=nii_dir, plot_dir=plot_dir)
        for key in comp:
            print("Running analyses on %s" % key)
            imgs.append(load_or_generate_components(
                    hemi=key, force=force, random_state=random_state, **kwargs))

        # Show confusion matrix:
        score_mat = compare_components(images=imgs, labels=comp,
                                   scoring=scoring)
        for normalize in [False, True]:
            plot_comparison_matrix(score_mat, scoring=scoring, normalize=normalize,
                               out_dir=plot_dir, keys=comp)

        # Show component comparisons
        plot_component_comparisons(images=imgs, labels=comp,
                               score_mat=score_mat, out_dir=plot_dir)

    return imgs, comp, score_mat


if __name__ == '__main__':
    import warnings
    from argparse import ArgumentParser

    # Look for image computation errors
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    # Arg parsing
    parser = ArgumentParser(description="Run ICA on individual hemispheres, "
                                        "or whole brain, then compare.\n\n"
                                        "R=right-only, L=left-only,\n"
                                        "RL=R,L ICA separate, compare as one\n"
                                        "wb=ICA & compare together")
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--offline', action='store_true', default=False)
    parser.add_argument('--qc', action='store_true', default=False)
    parser.add_argument('--components', nargs='?', type=int, default=20,
                        dest='n_components')
    parser.add_argument('--dataset', nargs='?', default='neurovault',
                        choices=['neurovault', 'abide', 'nyu'])
    parser.add_argument('--seed', nargs='?', type=int, default=42,
                        dest='random_state')
    parser.add_argument('--scoring', nargs='?', default='l1norm',
                        choices=['l1norm', 'l2norm', 'correlation'])
    args = vars(parser.parse_args())

    # Run qc
    query_server = not args.pop('offline')
    if args.pop('qc'):
        from qc import qc_image_data
        qc_image_data(args['dataset'], query_server=query_server)

    # Run main
    main(query_server=query_server, **args)

    plt.show()
