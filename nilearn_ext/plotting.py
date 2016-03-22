# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD

import os
import os.path as op

import numpy as np
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn.image import iter_img, index_img, new_img_like, math_img
from nilearn.plotting import plot_stat_map
from scipy import stats

from nilearn_ext.utils import reorder_mat


def save_and_close(out_path, fh=None):
    fh = fh or plt.gcf()
    if not op.exists(op.dirname(out_path)):
        os.makedirs(op.dirname(out_path))
    fh.savefig(out_path)
    plt.close(fh)


def _title_from_terms(terms, ic_idx, label=None, n_terms=4, flip_sign=False):

    if terms is None:
        return '%s[%d]' % (label, ic_idx)

    # Use the 4 terms weighted most as a positive title, 4 terms 
    # weighted least as a negative title and return both
    
    ica_terms = np.asarray(terms.values()).T
    ic_terms = ica_terms[ic_idx]
    terms = np.asarray(terms.keys())
    
    if flip_sign:
        positive_terms = terms[np.argsort(ic_terms)[:n_terms]]
        negative_terms = terms[np.argsort(ic_terms)[:-(n_terms+1):-1]]
    else:   
        positive_terms = terms[np.argsort(ic_terms)[:-(n_terms+1):-1]]
        negative_terms = terms[np.argsort(ic_terms)[:n_terms]]
    title = '%s[%d]: POS(%s) \n NEG(%s)' % (
        label, ic_idx, ', '.join(positive_terms),', '.join(negative_terms))
    
    return title


def plot_components(ica_image, hemi='', out_dir=None,
                    bg_img=datasets.load_mni152_template()):
    print("Plotting %s components..." % hemi)

    for ci, ic_img in enumerate(iter_img(ica_image)):
        # Threshhold and title
        # get nonzero part of the image for proper thresholding of
        # r- or l- only component
        nonzero_img = ic_img.get_data()[np.nonzero(ic_img.get_data())]
        ic_thr = stats.scoreatpercentile(np.abs(nonzero_img), 90)
        title = _title_from_terms(terms=ica_image.terms, ic_idx=ci, label=hemi)
        fh = plt.figure(figsize=(14, 6))
        plot_stat_map(ic_img, axes=fh.gca(), threshold=ic_thr, colorbar=True,
                      title=title, black_bg=True, bg_img=bg_img)

        # Save images instead of displaying
        if out_dir is not None:
            save_and_close(out_path=op.join(
                out_dir, '%s_component_%i.png' % (hemi, ci)))


def plot_component_comparisons(images, labels, score_mat, sign_mat, out_dir=None):
    """ 
    Uses the score_mat to find the closest components of each image, then plots 
    them side-by-side with equal colorbars. It finds the best match for every reference 
    component (on y-axis of score-mat), i.e. it allows non-one-to-one matching. 
    
    The sign_mat is used to check the sign flipping of the component comparisons, 
    and flips the matching component if necessary.
    
    If there are any unmatched component (on x-axis) it also plots them with their 
    best-matching ref component. 
    """
    # Be careful
    assert len(images) == 2
    assert len(labels) == 2
    assert images[0].shape == images[1].shape
    n_components = images[0].shape[3]  # values @ 0 and 1 are the same

    # Find cross-image mapping...Note that this allows non-one-to-one matching
    # i.e. for every given reference component (on y-axis) the best matching component
    # is chosen, even if that component has been chosen before
    most_similar_idx = score_mat.argmin(axis=1)
    
    # Keep track of unmatched components (on x-axis)
    unmatched = np.setdiff1d(np.arange(n_components), most_similar_idx)
    unmatched_msi = score_mat.argmin(axis=0)
    
    print("Plotting results.")
    for c1i in range(n_components+len(unmatched)):
        
        if c1i < n_components:
            cis = [c1i, most_similar_idx[c1i]]
            png_name = '%s_%s_%s.png' % (labels[0], labels[1], c1i)
       
        # plot leftover components, matched to their closest ref component
        else:
            umi = unmatched[c1i-n_components]
            cis = [unmatched_msi[umi], umi]
            png_name = 'unmatched_%s_%s.png' % (labels[1], c1i - n_components) 
        
        comp_imgs = [index_img(img, ci) for img, ci in zip(images, cis)] 
        
        # flip the sign if sign_mat for the corresponding comparison is -1
        if sign_mat[cis[0],cis[1]] == -1:
            comp_imgs[1] = math_img("-img", img = comp_imgs[1]) 
            flip_signs = (False, True)
        else:
            flip_signs = (False, False)  
        dat = [img.get_data() for img in comp_imgs]

        if ('R' in labels and 'L' in labels):
            # Combine left and right image, show just one.
            # need to think about what to do with the terms here when flipping one image..
            comp = new_img_like(comp_imgs[0], data=np.sum(dat, axis=0),
                                copy_header=True)
            title = _title_from_terms(terms=comp.terms, ic_idx=cis[1],
                                      label='R[%d] vs. L' % cis[0])
            fh = plt.figure(figsize=(14, 6))
            plot_stat_map(comp, axes=fh.gca(), title=title, black_bg=True)

        else:
            # Show two images, one above the other.

            vmax = np.abs(np.asarray(dat)).max()  # Determine scale bars
            fh = plt.figure(figsize=(14, 12))

            for ii in [0, 1]:  # Subplot per image
                ax = fh.add_subplot(2, 1, ii + 1)
                comp = comp_imgs[ii]
                
                title = _title_from_terms(terms=images[ii].terms, ic_idx=cis[ii], 
                                    label=labels[ii], flip_sign = flip_signs[ii])

                if ii == 0:
                    display = plot_stat_map(comp, axes=ax, title=title,    # noqa
                                            black_bg=True, symmetric_cbar=True,
                                            vmax=vmax)
                else:
                    # use same cut coords
                    cut_coords = display.cut_coords  # noqa
                    display = plot_stat_map(comp, axes=ax, title=title,
                                            black_bg=True, symmetric_cbar=True, 
                                            vmax=vmax, display_mode='ortho',
                                            cut_coords=cut_coords)

        # Save images instead of displaying
        if out_dir is not None:
            save_and_close(out_path=op.join(out_dir, png_name), fh=fh)


def plot_comparison_matrix(score_mat, scoring, normalize=True, out_dir=None,
                           keys=('R', 'L'), vmax=None, colorbar=True):

    # Settings
    score_mat, x_idx, y_idx = reorder_mat(score_mat, normalize=normalize)
    idx = np.arange(score_mat.shape[0])
    vmax = vmax  # or min(scores.max(), 10 if normalize else np.inf)
    vmin = 0  # 1 if normalize else 0

    # Plotting
    fh = plt.figure(figsize=(10, 10))
    ax = fh.gca()
    cax = ax.matshow(score_mat, vmin=vmin, vmax=vmax)
    ax.set_xlabel("%s components" % (keys[1]))
    ax.set_ylabel("%s components" % (keys[0]))
    ax.set_xticks(idx), ax.set_xticklabels(x_idx)
    ax.set_yticks(idx), ax.set_yticklabels(y_idx)
    if colorbar:
        fh.colorbar(cax)

    # Saving
    if out_dir is not None:
        save_and_close(out_path=op.join(out_dir, '%s_%s_simmat%s.png' % (
            keys[0], keys[1], '-normalized' if normalize else '')))
