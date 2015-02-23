import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_in_grid(imgs, plot_fn, aspect_ratio=1.3333, shape=None, figure=None,
                 **kwargs):
    """
    """
    if not isinstance(imgs, collections.Iterable):
        return plot_in_grid([imgs], plot_fn=plot_fn, aspect_ratio=aspect_ratio,
                            figure=figure, shape=shape, **kwargs)

    # Compute the grid size, and validate
    n_plots = len(imgs)
    if shape is None:
        n_rows = int(np.round(np.sqrt(n_plots / aspect_ratio)))
        n_cols = int(np.ceil(n_plots / n_rows))
        shape = (n_rows, n_cols)
    if n_plots > np.asarray(shape).prod():
        raise ValueError("Not enough subplots to accommodate images.")

    # Massage params
    def ensure_list(val, max_iters):
        if not isinstance(val, collections.Iterable):
            return list(itertools.repeat(val, max_iters))
        elif len(val) == max_iters:
            return list(val)
        else:
            raise ValueError("Lengths don't match")

    plot_fn = ensure_list(plot_fn, n_plots)
    for arg_name, arg_val in kwargs.items():
        kwargs[arg_name] = ensure_list(arg_val, n_plots)

    # Plot the grid.
    figure = figure or plt.figure()
    for fi in range(n_plots):
        ax = figure.add_subplot(fi, n_rows, n_cols)
        #plot_args = dict([(next(arg), next(val[fi]) for arg, val in kwargs.items()])
        plot_fn(axes=ax)#, **plot_args)

    return figure
