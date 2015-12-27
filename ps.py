# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD
"""
How many ics? Find the sweet spot where the
average dissimilarity between unilateral ('RL') and
bilateral ('both') components is minimized.

Do that with a hard loop on the # of components, then
plotting the mean dissimilarity (derived from the
score matrix).
"""

import os.path as op

import matplotlib.pyplot as plt

from main import main
from nilearn_ext.plotting import save_and_close


def do_ic_loop(components, scoring, dataset, **kwargs):
    score_mats = []
    for c in components:
        print("Running analysis with %d components." % c)
        score_mats.append(
            main(n_components=c, dataset=dataset, scoring=scoring,
                 **kwargs)[2])

    # Now we have all the scores; now plot.
    out_dir = op.join('ica_imgs', dataset)
    out_path = op.join(out_dir, '%s.png' % scoring)
    plt.plot(components, [sm.mean() for sm in score_mats])
    save_and_close(out_path)


if __name__ == '__main__':
    import warnings
    from argparse import ArgumentParser

    # Look for image computation errors
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    # Arg parsing
    hemi_choices = ['R', 'L', 'both']
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
