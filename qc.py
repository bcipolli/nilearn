"""
This script helps visualize brains, to validate
what looks good and what looks like crap.
"""

import os.path as op
import shutil

import matplotlib.pyplot as plt
import numpy as np
from nilearn.plotting import plot_stat_map
from sklearn.externals.joblib import Memory

from nilearn_ext.datasets import fetch_neurovault
from nilearn_ext.image import clean_img, cast_img
from nilearn_ext.masking import MniNiftiMasker
from nilearn_ext.plotting import save_and_close


def qc_image_metadata(**kwargs):
    images = fetch_neurovault(fetch_terms=False, **kwargs)[0]

    print len(images)

    for key in sorted(images[0].keys()):
        unique_vals = np.unique([im.get(key, 'blue') for im in images])
        print("%s (%d): " % (key, len(unique_vals)),
              unique_vals[:5])
        print("Sample image with missing value",
              [im['url'] for im in images if key not in im][-1:])
        print("")


def qc_image_data(**kwargs):
    # Download matching images
    images = fetch_neurovault(fetch_terms=False, **kwargs)[0]
    plot_dir = 'qc'

    # Get ready
    masker = MniNiftiMasker(memory=Memory(cachedir='nilearn_cache')).fit()
    if op.exists(plot_dir):  # Delete old plots.
        shutil.rmtree(plot_dir)

    for ii, image in enumerate(images):
        ri = ii % 4  # row i
        ci = (ii / 4) % 4  # column i
        pi = ii % 16 + 1  # plot i
        fi = ii / 16  # figure i

        if ri == 0 and ci == 0:
            fh = plt.figure(figsize=(16, 10))
            print('Plot %03d of %d' % (fi + 1, np.ceil(len(images) / 16.)))
        ax = fh.add_subplot(4, 4, pi)
        title = op.join(str(image['collection_id']), str(image['id']),
                        op.basename(image['local_path']))

        # Images may fail to be transformed, and are of different shapes,
        # so we need to trasnform one-by-one and keep track of failures.
        img = cast_img(image['local_path'], dtype=np.float32)
        img = clean_img(img)
        try:
            img = masker.inverse_transform(masker.transform(img))
        except Exception as e:
            print("Failed to mask/reshape image %s: %s" % (title, e))

        plot_stat_map(img, axes=ax, black_bg=True, title=title, colorbar=False)

        if (ri == 3 and ci == 3) or ii == len(images) - 1:
            out_path = op.join('qc', 'fig%03d.png' % (fi + 1))
            save_and_close(out_path)


if __name__ == '__main__':
    import warnings
    from argparse import ArgumentParser

    # Look for image computation errors
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    # Arg parsing
    parser = ArgumentParser(description="Really?")
    parser.add_argument('check', nargs='?', default='data',
                        choices=('data', 'metadata'))
    parser.add_argument('--offline', action='store_true', default=False)
    args = vars(parser.parse_args())

    # Alias args
    query_server = not args.pop('offline')
    if args.pop('check') == 'data':
        qc_image_data(query_server=query_server, **args)
    else:
        qc_image_metadata(query_server=query_server, **args)

    plt.show()
