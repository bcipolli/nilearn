# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD

import os
import os.path as op

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn.image import iter_img, index_img, new_img_like, math_img
from nilearn.plotting import plot_stat_map
from scipy import stats

from nilearn_ext.utils import reorder_mat, get_ic_terms, get_n_terms
from nilearn_ext.radar import radar_factory

import math

def nice_number(value, round_=False):
    '''nice_number(value, round_=False) -> float'''
    exponent = math.floor(math.log(value, 10))
    fraction = value / 10 ** exponent

    if round_:
        if fraction < 1.5: nice_fraction = 1.
        elif fraction < 3.: nice_fraction = 2.
        elif fraction < 7.: nice_fraction = 5.
        else: nice_fraction = 10.
    else:
        if fraction <= 1: nice_fraction = 1.
        elif fraction <= 2: nice_fraction = 2.
        elif fraction <= 5: nice_fraction = 5.
        else: nice_fraction = 10.

    return nice_fraction * 10 ** exponent

def nice_bounds(axis_start, axis_end, num_ticks=8):
    '''
    nice_bounds(axis_start, axis_end, num_ticks=10) -> tuple
    @return: tuple as (nice_axis_start, nice_axis_end, nice_tick_width)
    '''
    axis_width = axis_end - axis_start
    if axis_width == 0:
        nice_tick_w = 0
    else:
        nice_range = nice_number(axis_width)
        nice_tick_w = nice_number(nice_range / (num_ticks -1), round_=True)
        axis_start = math.floor(axis_start / nice_tick_w) * nice_tick_w
        axis_end = math.ceil(axis_end / nice_tick_w) * nice_tick_w
 
    nice_tick = np.arange(axis_start, axis_end, nice_tick_w)[1:]
    return axis_start, axis_end, nice_tick

def save_and_close(out_path, fh=None):
    fh = fh or plt.gcf()
    if not op.exists(op.dirname(out_path)):
        os.makedirs(op.dirname(out_path))
    fh.savefig(out_path)
    plt.close(fh)
    
def _title_from_terms(terms, ic_idx, label=None, n_terms=4, flip_sign=False):

    if terms is None:
        return '%s[%d]' % (label, ic_idx)

    # Use the n terms weighted most as a positive title, n terms 
    # weighted least as a negative title and return both
    
    pos_terms = get_n_terms(terms, ic_idx, n_terms=n_terms, flip_sign=flip_sign)
    neg_terms = get_n_terms(terms, ic_idx, n_terms=n_terms, top_bottom="bottom", 
                                    flip_sign=flip_sign)
                                    
    title = '%s[%d]: POS(%s) \n NEG(%s)' % (
        label, ic_idx, ', '.join(pos_terms),', '.join(neg_terms))
    
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
                
def plot_components_summary(ica_image, hemi='', out_dir=None,
                    bg_img=datasets.load_mni152_template()):
    print("Plotting %s components summary..." % hemi)
    
    n_components = ica_image.get_data().shape[3]
    for ii, ic_img in enumerate(iter_img(ica_image)):
        
        ri = ii % 5  # row i
        ci = (ii / 5) % 5  # column i
        pi = ii % 25 + 1  # plot i
        fi = ii / 25  # figure i

        if ri == 0 and ci == 0:
            fh = plt.figure(figsize=(30, 20))
            print('Plot %03d of %d' % (fi + 1, np.ceil(n_components / 25.)))
        ax = fh.add_subplot(5, 5, pi)
        
        # Threshhold and title
        # get nonzero part of the image for proper thresholding of
        # r- or l- only component
        nonzero_img = ic_img.get_data()[np.nonzero(ic_img.get_data())]
        ic_thr = stats.scoreatpercentile(np.abs(nonzero_img), 90)
        title = _title_from_terms(terms=ica_image.terms, ic_idx=ii, label=hemi)
        
        plot_stat_map(ic_img, axes=ax, threshold=ic_thr, colorbar=True,
                      title=title, black_bg=True, bg_img=bg_img)
                      
        if (ri == 4 and ci == 4) or ii == n_components - 1:
            out_path = op.join(out_dir, '%s_components_summary%02d.png' % (hemi, fi + 1))
            save_and_close(out_path)


def plot_component_comparisons(images, labels, score_mat, sign_mat, out_dir=None):
    """ 
    Uses the score_mat to find the closest components of each image, then plots 
    them side-by-side with equal colorbars. It finds the best match for every reference 
    component (on y-axis of score-mat), i.e. it allows non-one-to-one matching. 
    
    The sign_mat is used to check the sign flipping of the component comparisons, 
    and flips the matching component if necessary.
    
    If there are any unmatched component (on x-axis) it also plots them with their 
    best-matching ref component. 
    
    It plots and saves component comparisons and returns matched and unmatched indices.
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
            # terms are not combined here
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

    return most_similar_idx, unmatched

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


def plot_term_comparisons(label_list, term_list, ic_idx_list, sign_list, color_list,
                        top_n=4, bottom_n=4, standardize=True, out_dir=None):
    '''
    Take the list of ica image terms and the indices of components to be compared, and 
    plots the top n and bottom n term values for each component as a radar graph.
    
    The sign_list should indicate whether term values should be flipped (-1) or not (1).
    '''
    assert len(term_list)==len(label_list)
    assert len(term_list)==len(ic_idx_list)
    assert len(term_list)==len(sign_list)
    assert len(term_list)==len(color_list)
    
    terms_of_interest =[]
    term_vals = []
    name = ''

    for i, term, sign, label in zip(ic_idx_list, term_list, sign_list, label_list):
        flip_sign = True if sign == -1 else False
        
        # Get list of top n and bottom n terms for each term list  
        top_terms = get_n_terms(term, i, n_terms=top_n, top_bottom = 'top', flip_sign=flip_sign)
        bottom_terms = get_n_terms(term, i, n_terms=bottom_n, top_bottom = 'bottom', flip_sign=flip_sign)
        combined = np.append(top_terms, bottom_terms)
        terms_of_interest.append(combined)
        
        # Also store term vals (z-score if standardize) for each list
        terms, vals = get_ic_terms(term, i, flip_sign = flip_sign, standardize = standardize)
        s = pd.Series(vals, index = terms, name = label)
        term_vals.append(s)
        
        # Construct name for the comparison
        name += label + '[%d] '%(i)
          
        
    term_df = pd.concat(term_vals, axis = 1)
    
    # Get unique terms from terms_of_interest list
    toi_unique = np.unique(terms_of_interest)
    
    # Get values for unique terms_of_interest
    data = term_df.loc[toi_unique]
    data = data.sort_values(label_list, ascending =False)  
    
    # Now plot radar!
    N = len(toi_unique)
    theta = radar_factory(N)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1,1,1, projection='radar')
    title = "Term comparisons for %scomponents"%(name)
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')
    
    y_min, y_max, y_tick  = nice_bounds(data.values.min(), data.values.max())
    print y_min, y_max, y_tick
    ax.set_ylim(y_min,y_max)
    ax.set_yticks([0], minor=True)
    ax.set_yticks(y_tick)
    ax.yaxis.grid(which='major', linestyle=':')
    ax.yaxis.grid(which='minor', linestyle='-') 
         
    for label, color in zip(label_list, color_list):
        ax.plot(theta,data[label], color = color)
        ax.fill(theta,data[label], facecolor=color, alpha=0.25) 
    ax.set_varlabels(data.index.values)
    
    legend = plt.legend(label_list, loc=(1.1, 0.9), labelspacing=0.1)
    plt.setp(legend.get_texts(), fontsize='small')
    plt.show()
    return data