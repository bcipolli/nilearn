plotted_subject = 0  # subject to plot
n_jobs = 1

import numpy as np

import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import datasets
from nilearn.image import iter_img, reorder_img, new_img_like
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
from six import string_types
from sklearn.externals.joblib import Memory


def load_if_needed(img):
    """Convenience function to load image if a string path"""
    if isinstance(img, string_types):
        img = nib.load(img)
    return img


def flip_img_lr(img):
    """ Convenience function to flip image on X axis"""
    img = load_if_needed(img)
    # This won't work for all image formats! But
    # does work for those that we're working with...
    assert isinstance(img, nib.nifti1.Nifti1Image)
    img = new_img_like(img, data=img.get_data()[::-1], copy_header=True)
    return img


def split_bilateral_rois(maps_img, show_results=False):
    """Convenience function for splitting bilateral ROIs
    into two unilateral ROIs"""

    new_rois = []

    for map_img in iter_img(maps_img):
        if show_results:
            plot_stat_map(map_img, title='raw')
        for hemi in ['L', 'R']:
            hemi_mask = HemisphereMasker(hemisphere=hemi)
            hemi_mask.fit(map_img)
            if hemi_mask.mask_img_.get_data().sum() > 0:
                hemi_vectors = hemi_mask.transform(map_img)
                hemi_img = hemi_mask.inverse_transform(hemi_vectors)
                new_rois.append(hemi_img.get_data())
                if show_results:
                    plot_stat_map(hemi_img, title=hemi)
        if show_results:
            plt.show()
    new_maps_data = np.concatenate(new_rois, axis=3)
    new_maps_img = new_img_like(maps_img, data=new_maps_data)
    print ("Changed from %d ROIs to %d ROIs" % (maps_img.shape[-1],
                                                new_maps_img.shape[-1]))
    return new_maps_img


class HemisphereMasker(NiftiMasker):
    """
    Masker to segregate by hemisphere.

    Parameters
    ==========
    hemisphere: L or R

    """
    def __init__(self, mask_img=None, sessions=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background',
                 mask_args=None, sample_mask=None,
                 memory_level=1, memory=Memory(cachedir=None),
                 verbose=0, hemisphere='L'):
        if hemisphere.lower() in ['l', 'left']:
            self.hemi = 'l'
        elif hemisphere.lower() in ['r', 'right']:
            self.hemi = 'r'
        else:
            raise ValueError('Hemisphere must be left or right; '
                             'got value %s' % self.hemi)

        super(HemisphereMasker, self).__init__(mask_img=mask_img,
                                               sessions=sessions,
                                               smoothing_fwhm=smoothing_fwhm,
                                               standardize=standardize,
                                               detrend=detrend,
                                               low_pass=low_pass,
                                               high_pass=high_pass,
                                               t_r=t_r,
                                               target_affine=target_affine,
                                               target_shape=target_shape,
                                               mask_strategy=mask_strategy,
                                               mask_args=mask_args,
                                               sample_mask=sample_mask,
                                               memory_level=memory_level,
                                               memory=memory,
                                               verbose=verbose)

    def fit(self, X=None, y=None):  # noqa
        super(HemisphereMasker, self).fit(X, y)

        # x, y, z
        hemi_mask_data = reorder_img(self.mask_img_).get_data().astype(np.bool)

        xvals = hemi_mask_data.shape[0]
        midpt = np.ceil(xvals / 2.)
        if self.hemi == 'r':
            other_hemi_slice = slice(midpt, xvals)
        else:
            other_hemi_slice = slice(0, midpt)

        hemi_mask_data[other_hemi_slice] = False
        mask_data = self.mask_img_.get_data() * hemi_mask_data
        self.mask_img_ = new_img_like(self.mask_img_, data=mask_data)

        return self


class MniHemisphereMasker(HemisphereMasker):
    """ Alias for HemisphereMasker with mask_img==Mni template"""
    def __init__(self, sessions=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background',
                 mask_args=None, sample_mask=None,
                 memory_level=1, memory=Memory(cachedir=None),
                 verbose=0, hemisphere='L'):
        target_img = datasets.load_mni152_template()
        grey_voxels = (target_img.get_data() > 0).astype(int)
        mask_img = new_img_like(target_img, grey_voxels)

        super(MniHemisphereMasker, self).__init__(
            mask_img=mask_img,
            sessions=sessions,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            detrend=detrend,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            target_affine=target_affine,
            target_shape=target_shape,
            mask_strategy=mask_strategy,
            mask_args=mask_args,
            sample_mask=sample_mask,
            memory_level=memory_level,
            memory=memory,
            verbose=verbose,
            hemisphere=hemisphere)

    def mask_as_img(self, img):
        """ Convenience function to mask image, return as image."""
        X = self.fit_transform(img)  # noqa
        return self.inverse_transform(X)
