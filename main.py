import os.path as op

import numpy as np

from compare_components import (compare_components,
                                prep_image_for_comparison)
from generate_components import (download_images_and_terms,
                                 generate_components,
                                 plot_components)
from nifti_with_terms import NiftiImageWithTerms


def main(keys=('R', 'L'), n_components=20, n_images=np.inf,
         force=False, img_dir=None, plot_dir=None):
    img_dir = img_dir or op.join('ica_nii', str(n_components))
    plot_dir = plot_dir or op.join('ica_maps', str(n_components))

    # Download
    images, term_scores = download_images_and_terms(n_images=n_images)

    # Analyze
    print("Running all analyses on both hemis together, and each separately.")
    imgs = dict()
    for hemi in [key1, key2]:
        # Only re-run if image doesn't exist.
        img_path = op.join(img_dir, '%s_ica_components.nii.gz' % hemi)
        if not force and op.exists(img_path):
            imgs[hemi] = NiftiImageWithTerms.from_filename(img_path)
        else:
            imgs[hemi] = generate_components(
                images=images, term_scores=term_scores,
                hemi=hemi, n_components=n_components,
                out_dir=img_dir)

        plot_components(imgs[hemi], hemi=hemi, out_dir=plot_dir)

    # Get the requested images
    for key in imgs:
        imgs[key] = prep_image_for_comparison(key, imgs[key])
    compare_components(images=imgs.values(), labels=imgs.keys())


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
