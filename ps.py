# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD
"""
How many ics? Find the sweet spot where the
average dissimilarity between unilateral ('RL') and
bilateral ('wb') components is minimized.

Do that with a hard loop on the # of components, then
plotting the mean dissimilarity (derived from the
score matrix).
"""

import os.path as op

import matplotlib.pyplot as plt
import numpy as np

from main import main
from nilearn_ext.plotting import save_and_close


def do_ic_loop(components, scoring, dataset, **kwargs):
    score_mats = []
    images = []
    for c in components:
        print("Running analysis with %d components." % c)
        imgs, lbls, sm = main(n_components=c, dataset=dataset,
                              scoring=scoring, **kwargs)
        score_mats.append(sm)
        images.append(imgs[0])

    # Now we have all the scores; now plot.
    lines = dict(err=[], above_0005=[], above_001=[], above_005=[])
    for sm, img in zip(score_mats, images):
        sm[sm == 0] = np.inf
        data = img.get_data()[img.get_data() > 0]

        # we want to maximize how similar the most confusible components are.

        lines['err'].append(sm.min(axis=1).mean())

        # Track how sparse the features are.
        lines['above_0005'].append(np.abs(data > 0.005).sum())
        lines['above_005'].append(np.abs(data > 0.05).sum())
        lines['above_001'].append(np.abs(data > 0.01).sum())

    # Normalize
    for key in lines:
        total = float(max(1E-10, np.sum(lines[key])))
        lines[key] = np.asarray(lines[key]) / total

    fh = plt.figure(figsize=(10, 10))
    fh.gca().plot(components, np.asarray(lines.values())[-1].T)
    # fh.gca().legend(lh, lines.keys())

    out_dir = op.join('ica_imgs', dataset)
    out_path = op.join(out_dir, '%s.png' % scoring)
    save_and_close(out_path, fh=fh)


if __name__ == '__main__':
    import warnings
    from argparse import ArgumentParser

    # Look for image computation errors
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    # Arg parsing
    hemi_choices = ['R', 'L', 'wb']
    parser = ArgumentParser(description="Really?")
    parser.add_argument('key1', nargs='?', default='R', choices=hemi_choices)
    parser.add_argument('key2', nargs='?', default='L', choices=hemi_choices)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--offline', action='store_true', default=False)
    parser.add_argument('--components', nargs='?',
                        default="5,10,15,20,25,30,35,40,45,50")
    parser.add_argument('--dataset', nargs='?', default='neurovault',
                        choices=['neurovault', 'abide', 'nyu'])
    parser.add_argument('--seed', nargs='?', type=int, default=42,
                        dest='random_state')
    parser.add_argument('--scoring', nargs='?', default='scoring',
                        choices=['l1norm', 'l2norm', 'correlation'])
    args = vars(parser.parse_args())

    # Alias args
    query_server = not args.pop('offline')
    keys = args.pop('key1'), args.pop('key2')
    components = [int(c) for c in args.pop('components').split(',')]

    # Run loops
    do_ic_loop(keys=keys, query_server=query_server,
               components=components, **args)

    plt.show()
