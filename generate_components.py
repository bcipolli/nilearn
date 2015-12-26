# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsuchida
# License: BSD

import os
import os.path as op

import numpy as np
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn.image import new_img_like, iter_img
from nilearn.input_data import NiftiMasker
from nilearn._utils import check_niimg
from nilearn.plotting import plot_stat_map
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.externals.joblib import Memory

from hemisphere_masker import MniHemisphereMasker
from nifti_with_terms import NiftiImageWithTerms


def clean_img(img):
    """ Remove nan/inf entries."""
    img = check_niimg(img)
    img_data = img.get_data()
    img_data[np.isnan(img_data)] = 0
    img_data[np.isinf(img_data)] = 0
    return new_img_like(img, img_data, copy_header=True)


def cast_img(img, dtype=np.float32):
    """ Cast image to the specified dtype"""
    img = check_niimg(img)
    img_data = img.get_data().astype(dtype)
    return new_img_like(img, img_data, copy_header=True)


def download_images_and_terms(n_images=np.inf):
    # Get image and term data

    # Download 100 matching images
    ss_all = datasets.fetch_neurovault(max_images=n_images,
                                       map_types=['F map', 'T map', 'Z map'],
                                       fetch_terms=True)
    images = ss_all['images']
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
    term_matrix = term_matrix[:, xformable_idx]

    # Run ICA and map components to terms
    print("%s: Running ICA; may take time..." % hemi)
    fast_ica = FastICA(n_components=n_components, random_state=random_state)
    ica_maps = memory.cache(fast_ica.fit_transform)(X.T).T

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


def plot_components(ica_image, hemi='', out_dir=None,
                    bg_img=datasets.load_mni152_template()):
    print("Plotting %s components..." % hemi)
    terms = np.asarray(ica_image.terms.keys())
    ica_terms = np.asarray(ica_image.terms.values()).T

    idx = 0
    for ic_img, ic_terms in zip(iter_img(ica_image), ica_terms):
        idx += 1
        ic_thr = stats.scoreatpercentile(np.abs(ic_img.get_data()), 90)
        display = plot_stat_map(ic_img, threshold=ic_thr, colorbar=False,
                                bg_img=bg_img)

        # Use the 4 terms weighted most as a title
        important_terms = terms[np.argsort(ic_terms)[-4:]]
        title = '%d: %s' % (idx, ','.join(important_terms[::-1]))
        display.title(title, size=16)

        # Save images instead of displaying
        if out_dir is not None:
            out_path = op.join(out_dir, '%s_component_%i.png' % (hemi, idx))
            if not op.exists(op.dirname(out_path)):
                os.makedirs(op.dirname(out_path))
            plt.savefig(out_path)
            plt.close()


if __name__ == '__main__':

    import warnings
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    n_components = 20
    img_dir = op.join('ica_nii', str(n_components))
    plot_dir = op.join('ica_maps', str(n_components))

    images, term_scores = download_images_and_terms()

    print("Running all analyses on both hemis together, and each separately.")
    for hemi in ['both', 'R', 'L']:
        img_path = op.join(img_dir, '%s_ica_components.nii.gz' % hemi)
        if not op.exists(img_path):
            generate_components(images=images, hemi=hemi,
                                n_components=20, out_dir=img_dir,
                                memory=Memory(cachedir='nilearn_cache'))

        plot_components(img_path)
