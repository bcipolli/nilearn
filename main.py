# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsuchida
# License: BSD

import os.path as op

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn.image import index_img, iter_img

from compare_components import (compare_components, plot_comparisons)
from generate_components import (download_images_and_terms,
                                 generate_components,
                                 plot_components)
from nifti_with_terms import NiftiImageWithTerms
from hemisphere_masker import join_bilateral_rois


def load_or_generate_components(hemi, out_dir='.', *args, **kwargs):
    # Only re-run if image doesn't exist.
    img_path = op.join(out_dir, '%s_ica_components.nii.gz' % hemi)
    if not kwargs.get('force') and op.exists(img_path):
        return NiftiImageWithTerms.from_filename(img_path)

    else:
        img = generate_components(hemi=hemi, out_dir=out_dir, *args, **kwargs)
        plot_components(img, hemi=hemi, out_dir=kwargs.get('plot_dir'))


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
        term_scores.append([(R_img.terms[t][rci] +
                             L_img.terms[t][lci]) / 2
                            for t in terms])

    # Squash into single image
    img = nib.concat_images(bilat_imgs)
    img.terms = dict(zip(terms, np.asarray(term_scores).T))
    return img


def main(keys=('R', 'L'), n_components=20, n_images=np.inf,
         force=False, img_dir=None, plot_dir=None):
    img_dir = img_dir or op.join('ica_nii', str(n_components))
    plot_dir = plot_dir or op.join('ica_imgs', str(n_components))

    # Download
    images, term_scores = download_images_and_terms(
        n_images=n_images, query_server=False)

    # Analyze
    print("Running all analyses on both hemis together, and each separately.")
    imgs = dict()
    for key in keys:
        kwargs = dict(images=images, term_scores=term_scores,
                      n_components=n_components, out_dir=img_dir)

        if key.lower() not in ('rl', 'lr'):
            imgs[key] = load_or_generate_components(hemi=key, **kwargs)

        else:
            imgs[key] = mix_and_match_bilateral_components(**kwargs)

    # Get the requested images
    score_mat = compare_components(images=imgs.values(), labels=imgs.keys())
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
    main(keys=(key1, key2), n_components=n_components, force=False)
    plt.show()
