# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsuchida
# License: BSD

import os
import os.path as op

import numpy as np
from nilearn import datasets
from nilearn.image import new_img_like, iter_img
from nilearn.input_data import NiftiMasker
from sklearn.decomposition import FastICA
from sklearn.externals.joblib import Memory

from nibabel_ext import NiftiImageWithTerms
from .masking import MniHemisphereMasker
from .utils import cast_img, clean_img


def generate_components(images, term_scores, hemi,
                        n_components=20, random_state=42,
                        out_dir=None, memory=Memory(cachedir='nilearn_cache')):
    terms = term_scores.keys()
    term_matrix = np.asarray(term_scores.values())
    term_matrix[term_matrix < 0] = 0

    # Create grey matter mask from mni template
    target_img = datasets.load_mni152_template()
    grey_voxels = (target_img.get_data() > 0).astype(int)
    mask_img = new_img_like(target_img, grey_voxels, copy_header=True)

    # Reshape & mask images
    print("%s: Reshaping and masking images; may take time." % hemi)
    if hemi == 'both':
        masker = NiftiMasker(mask_img=mask_img,
                             target_affine=target_img.affine,
                             target_shape=target_img.shape,
                             memory=memory)

    else:  # R and L maskers
        masker = MniHemisphereMasker(target_affine=target_img.affine,
                                     target_shape=target_img.shape,
                                     memory=memory,
                                     hemisphere=hemi)
    masker = masker.fit()

    # Images may fail to be transformed, and are of different shapes,
    # so we need to trasnform one-by-one and keep track of failures.
    X = []  # noqa
    xformable_idx = np.ones((len(images),), dtype=bool)
    for ii, im in enumerate(images):
        img = cast_img(im['local_path'], dtype=np.float32)
        img = clean_img(img)
        try:
            X.append(masker.transform(img))
        except Exception as e:
            print("Failed to mask/reshape image %d/%s: %s" % (
                im.get('collection_id', 0),
                op.basename(im['local_path']),
                e))
            xformable_idx[ii] = False

    # Now reshape list into 2D matrix, and remove failed images from terms
    X = np.vstack(X)  # noqa
    term_matrix = term_matrix[:, xformable_idx]  # terms x images

    # Run ICA and map components to terms
    print("%s: Running ICA; may take time..." % hemi)
    fast_ica = FastICA(n_components=n_components, random_state=random_state)
    fast_ica = memory.cache(fast_ica.fit)(X.T)
    ica_maps = memory.cache(fast_ica.transform)(X.T).T

    # Don't use the transform method as it centers the data
    ica_terms = np.dot(term_matrix, fast_ica.components_.T).T

    # Pretty up the results
    for idx, (ic, ic_terms) in enumerate(zip(ica_maps, ica_terms)):
        if -ic.min() > ic.max():
            # Flip the map's sign for prettiness
            ica_maps[idx] = -ic
            ica_terms[idx] = -ic_terms

    # Generate figures
    print("%s: Generating figures." % hemi)

    # Create image from maps, save terms to the image directly
    ica_image = NiftiImageWithTerms.from_image(
        masker.inverse_transform(ica_maps))
    ica_image.terms = dict(zip(terms, ica_terms.T))

    # Write to disk
    if out_dir is not None:
        out_path = op.join(out_dir, '%s_ica_components.nii.gz' % hemi)
        if not op.exists(op.dirname(out_path)):
            os.makedirs(op.dirname(out_path))
        ica_image.to_filename(out_path)
    return ica_image


def compare_components(images, labels):
    assert len(images) == 2
    assert len(labels) == 2
    assert images[0].shape == images[1].shape
    n_components = images[0].shape[3]  # values @ 0 and 1 are the same

    print("Loading images.")
    for img in images:
        img.get_data()  # Just loaded to get them in memory..

    print("Scoring closest components (by L1 norm)")
    score_mat = np.zeros((n_components, n_components))
    for c1i, comp1 in enumerate(iter_img(images[0])):
        for c2i, comp2 in enumerate(iter_img(images[1])):
            if 'R' in labels or 'L' in labels:
                hemi_idx = labels.index('R') or labels.index('L')
                masker = MniHemisphereMasker(hemisphere=labels[hemi_idx]).fit()
                c1_data = masker.transform(comp1)
                c2_data = masker.transform(comp2)
            else:
                c1_data = comp1.get_data()
                c2_data = comp2.get_data()
            l1norm = np.abs(c1_data - c2_data).sum()
            score_mat[c1i, c2i] = l1norm

    return score_mat
