# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsuchida
# License: BSD

import os
import os.path as op
import warnings

import numpy as np
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn.image import new_img_like
from nilearn.input_data import NiftiMasker
from nilearn._utils import check_niimg
from nilearn.plotting import plot_stat_map
from scipy import stats
from sklearn.decomposition import FastICA
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('error', RuntimeWarning)  # Catch numeric issues in imgs

from hemisphere_masker import HemisphereMasker


def clean_img(img):
    """ Remove nan/inf entries."""
    img = check_niimg(img)
    img_data = img.get_data()
    img_data[np.isnan(img_data)] = 0
    img_data[np.isinf(img_data)] = 0
    return new_img_like(img, img_data)


def cast_img(img, dtype=np.float32):
    """ Cast image to the specified dtype"""
    img = check_niimg(img)
    img_data = img.get_data().astype(dtype)
    return new_img_like(img, img_data)


# Get image and term data #####################################################
# Download 100 matching images
ss_all = datasets.fetch_neurovault(max_images=np.inf,  # Use np.inf for all.
                                   map_types=['F map', 'T map', 'Z map'],
                                   fetch_terms=True)
images, collections = ss_all['images'], ss_all['collections']
term_scores = ss_all['terms']

# Clean & report term scores
terms = np.array(term_scores.keys())
term_matrix = np.asarray(term_scores.values())
term_matrix[term_matrix < 0] = 0
total_scores = np.mean(term_matrix, axis=1)

print("Top 10 neurosynth terms from downloaded images:")
for term_idx in np.argsort(total_scores)[-10:][::-1]:
    print('\t%-25s: %.2f' % (terms[term_idx], total_scores[term_idx]))

print("Creating a grey matter mask.")
target_img = datasets.load_mni152_template()
grey_voxels = (target_img.get_data() > 0).astype(int)
mask_img = new_img_like(target_img, grey_voxels)

print("Running all analyses on both hemis together, and each separately.")
for hemi in ['both', 'R', 'L']:
    # Reshape & mask images ###################################################
    print("%s: Reshaping and masking images; may take time." % hemi)
    if hemi == 'both':
        masker = NiftiMasker(mask_img=mask_img,
                             target_affine=target_img.affine,
                             target_shape=target_img.shape,
                             memory='nilearn_cache')

    else:  # R and L maskers
        masker = HemisphereMasker(mask_img=mask_img,
                                  target_affine=target_img.affine,
                                  target_shape=target_img.shape,
                                  memory='nilearn_cache', hemisphere=hemi)
    masker = masker.fit()

    # Images may fail to be transformed, and are of different shapes,
    # so we need to trasnform one-by-one and keep track of failures.
    X = []
    xformable_idx = np.ones((len(images),), dtype=bool)
    for ii, im in enumerate(images):
        img = cast_img(im['local_path'], dtype=np.float32)
        img = clean_img(img)
        try:
            X.append(masker.transform(img))
        except Exception as e:
            print("Failed to mask/reshape image %d/%s: %s" % (
                im.get('collection_id', 0), op.basename(im['local_path']), e))
            xformable_idx[ii] = False

    # Now reshape list into 2D matrix, and remove failed images from terms
    X = np.vstack(X)
    term_matrix = term_matrix[:, xformable_idx]

    # Run ICA and map components to terms #####################################
    print("%s: Running ICA; may take time..." % hemi)
    fast_ica = FastICA(n_components=20, random_state=42)
    ica_maps = fast_ica.fit_transform(X.T).T

    # Don't use the transform method as it centers the data
    ica_terms = np.dot(term_matrix, fast_ica.components_.T).T

    # Generate figures ########################################################
    print("%s: Generating figures." % hemi)

    ica_images = masker.inverse_transform(ica_maps)

    # Write to disk
    out_path = op.join('ica_nii', '%s_ica_components.nii.gz' % hemi)
    if not op.exists(op.dirname(out_path)):
        os.makedirs(op.dirname(out_path))
    ica_images.to_filename(out_path)

    for idx, (ic, ic_terms) in enumerate(zip(ica_maps, ica_terms)):
        if -ic.min() > ic.max():
            # Flip the map's sign for prettiness
            ic = -ic
            ic_terms = -ic_terms

        ic_thr = stats.scoreatpercentile(np.abs(ic), 90)
        ic_img = masker.inverse_transform(ic)
        display = plot_stat_map(ic_img, threshold=ic_thr, colorbar=False,
                                bg_img=target_img)

        # Use the 4 terms weighted most as a title
        important_terms = terms[np.argsort(ic_terms)[-4:]]
        title = '%d: %s' % (idx, ', '.join(important_terms[::-1]))
        display.title(title, size=16)

        # Save images instead of displaying
        out_path = op.join('ica_maps', '%s_component_%i.png' % (hemi, idx))
        if not op.exists(op.dirname(out_path)):
            os.makedirs(op.dirname(out_path))
        plt.savefig(out_path)
        plt.close()
