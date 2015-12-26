"""
This script helps visualize brains, to validate
what looks good and what looks like crap.
"""

import os.path as op

import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets
from nilearn.plotting import plot_stat_map
from sklearn.externals.joblib import Memory

from nilearn_ext.image import clean_img, cast_img
from nilearn_ext.masking import MniNiftiMasker
from nilearn_ext.plotting import save_and_close

# Download matching images
ss_all = datasets.fetch_neurovault(max_images=np.inf,
                                   map_types=['F map', 'T map', 'Z map'],
                                   query_server=False,
                                   fetch_terms=False)
images = ss_all['images']
masker = MniNiftiMasker(memory=Memory(cachedir='nilearn_cache')).fit()

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
