"""
Functions relating to calculating and plotting posterior distributions.
"""

import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.special


def log_like_to_post(log_like, dx):
    """
    Convert log-likelihood to normalised posterior, while aiming to avoid over/underflow.

    Args:
        log_like (ND numpy array): N-dimensional grid of log-likelihood values.
        dx (float): Grid cell size, e.g. dx for 1D, dx*dy for 2D, etc.

    Returns:
        ND numpy array: Normalised posterior grid.
    """

    log_like -= np.amax(log_like) - 100 # to attempt to avoid over/underflow
    like = np.exp(log_like)
    post = like / (np.sum(like) * dx)
    assert np.isclose(np.sum(post) * dx, 1)
    return post


def post_to_conf(post_grid, cell_size):
    """
    Converts a N-dimensional grid of posterior values into a grid of confidence levels. The posterior values do not need
    to be normalised, i.e. their distribution need not integrate to 1. Works with likelihood values (not log-likelihood)
    instead of posteriors, assuming a flat prior.

    Args:
        post_grid (ND numpy array): Grid of posterior values.
        cell_size (float): The size of a grid cell, e.g. for 2 dimensions x and y this would be dx*dy.

    Returns:
        ND numpy array: Grid of confidence levels, where the value at each point is the minimum confidence region that \
                        includes that point. The least likely point would have a value of 1, indicating that it is \
                        only included in the 100% confidence region and excluded from anything smaller.
    """

    # Create flattened list of posteriors and sort in descending order
    posteriors = post_grid.flatten()
    posteriors[::-1].sort()

    # Dictionary to contain mapping between posterior and confidence level
    confidence_level_unnormalised = {}

    # Calculate the cumulative integral of posterior values
    integral = 0
    for posterior in posteriors:
        integral += posterior * cell_size
        confidence_level_unnormalised[posterior] = integral

    # Map each posterior in the grid to its confidence value
    confidence_grid_unnormalised = np.vectorize(confidence_level_unnormalised.get)(post_grid)

    # Normalise the confidence values using the final (complete) integral
    confidence_grid_normalised = np.divide(confidence_grid_unnormalised, integral)

    return confidence_grid_normalised


def grid_3d(log_like_path, save_path):
    """
    Calculate 3D posterior grids from a log-likelihood file containing two different likelihoods.

    Args:
        log_like_path (str): Path to log-likelihood file.
        save_path (str): Path to save posterior grid to.
    """

    # Load data
    data = np.loadtxt(log_like_path)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    log_like_1 = data[:, 3]
    log_like_2 = data[:, 4]

    # Convert log-likelihood to unnormalised posterior (assuming flat prior)
    log_like_1 -= np.amax(log_like_1) - 100 # to attempt to prevent over/underflow
    log_like_2 -= np.amax(log_like_2) - 100
    post_1 = np.exp(log_like_1)
    post_2 = np.exp(log_like_2)

    # Form grids
    x_range = np.unique(x)
    y_range = np.unique(y)
    z_range = np.unique(z)
    x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range, indexing='ij')

    # Grid the data
    post_grid_1 = scipy.interpolate.griddata((x, y, z), post_1, (x_grid, y_grid, z_grid), fill_value=0)
    post_grid_2 = scipy.interpolate.griddata((x, y, z), post_2, (x_grid, y_grid, z_grid), fill_value=0)

    # Save to file
    header = f'Output from {__file__}.grid_3d for input {log_like_path} at {time.strftime("%c")}'
    np.savez_compressed(save_path, x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, post_grid_1=post_grid_1,
                        post_grid_2=post_grid_2, header=header)
    print('Saved ' + save_path)


def plot_3d(grid_path, contour_levels_sig, smooth_sig_2d=0, smooth_sig_1d=0, like_label_1=None, like_label_2=None,
            x_label=None, y_label=None, z_label=None, x_lims=None, y_lims=None, z_lims=None, save_path=None):
    """
    Plot a 3D posterior grids produced by grid_3d as a single triangle plot.

    Args:
        grid_path (str): Path to 3D posterior grids as output by grid_3d.
        contour_levels_sig (list): List of sigma confidence regions to mark, e.g. [1, 2, 3].
        smooth_sig_2d (float, optional): Size of the 2D smoothing kernel in standard deviations.
        smooth_sig_1d (float, optional): Size of the 1D smoothing kernel in standard deviations.
        like_label_1 (str, optional): Legend label for the first likelihood.
        like_label_2 (str, optional): Legend label for the second likelihood.
        x_label (str, optional): x-axis label.
        y_label (str, optional): y-axis label.
        z_label (str, optional): z-axis label.
        x_lims ((float, float), optional): x-axis limits.
        y_lims ((float, float), optional): y-axis limits.
        z_lims ((float, float), optional): z-axis limits.
        save_path (str, optional): Path to save plot to, if supplied. If not supplied, plot will be displayed.
    """

    # Load the unnormalised 3D posteriors
    with np.load(grid_path) as data:
        x_grid = data['x_grid']
        y_grid = data['y_grid']
        z_grid = data['z_grid']
        post_grid_1 = data['post_grid_1']
        post_grid_2 = data['post_grid_2']

    # Form 1D & 2D grids
    x = x_grid[:, 0, 0]
    y = y_grid[0, :, 0]
    z = z_grid[0, 0, :]
    xy_x, xy_y = np.meshgrid(x, y, indexing='ij')
    xz_x, xz_z = np.meshgrid(x, z, indexing='ij')
    yz_y, yz_z = np.meshgrid(y, z, indexing='ij')

    # Calculate integration elements
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    assert np.allclose(np.diff(x), dx)
    assert np.allclose(np.diff(y), dy)
    assert np.allclose(np.diff(z), dz)
    dxdy = dx * dy
    dxdz = dx * dz
    dydz = dy * dz
    dxdydz = dx * dy * dz

    # Normalise the 3D posteriors
    post_grid_1 /= (np.sum(post_grid_1) * dxdydz)
    post_grid_2 /= (np.sum(post_grid_2) * dxdydz)
    assert np.isclose(np.sum(post_grid_1) * dxdydz, 1)
    assert np.isclose(np.sum(post_grid_2) * dxdydz, 1)

    # Marginalise to get 2D posteriors
    post_xy_1 = np.sum(post_grid_1, axis=2) * dz
    post_xy_2 = np.sum(post_grid_2, axis=2) * dz
    post_xz_1 = np.sum(post_grid_1, axis=1) * dy
    post_xz_2 = np.sum(post_grid_2, axis=1) * dy
    post_yz_1 = np.sum(post_grid_1, axis=0) * dx
    post_yz_2 = np.sum(post_grid_2, axis=0) * dx
    assert np.isclose(np.sum(post_xy_1) * dxdy, 1)
    assert np.isclose(np.sum(post_xy_2) * dxdy, 1)
    assert np.isclose(np.sum(post_xz_1) * dxdz, 1)
    assert np.isclose(np.sum(post_xz_2) * dxdz, 1)
    assert np.isclose(np.sum(post_yz_1) * dydz, 1)
    assert np.isclose(np.sum(post_yz_2) * dydz, 1)

    # Marginalise again to get 1D posteriors
    post_x_1 = np.sum(post_xy_1, axis=1) * dy
    post_x_2 = np.sum(post_xy_2, axis=1) * dy
    post_y_1 = np.sum(post_xy_1, axis=0) * dx
    post_y_2 = np.sum(post_xy_2, axis=0) * dx
    post_z_1 = np.sum(post_xz_1, axis=0) * dx
    post_z_2 = np.sum(post_xz_2, axis=0) * dx
    assert np.isclose(np.sum(post_x_1) * dx, 1)
    assert np.isclose(np.sum(post_x_2) * dx, 1)
    assert np.isclose(np.sum(post_y_1) * dy, 1)
    assert np.isclose(np.sum(post_y_2) * dy, 1)
    assert np.isclose(np.sum(post_z_1) * dz, 1)
    assert np.isclose(np.sum(post_z_2) * dz, 1)

    # Additional marginalisation checks
    assert np.allclose(post_x_1, np.sum(post_xz_1, axis=1) * dz)
    assert np.allclose(post_x_2, np.sum(post_xz_2, axis=1) * dz)
    assert np.allclose(post_y_1, np.sum(post_yz_1, axis=1) * dz)
    assert np.allclose(post_y_2, np.sum(post_yz_2, axis=1) * dz)
    assert np.allclose(post_z_1, np.sum(post_yz_1, axis=0) * dy)
    assert np.allclose(post_z_2, np.sum(post_yz_2, axis=0) * dy)
    assert np.allclose(post_x_1, np.sum(post_grid_1, axis=(1, 2)) * dydz)
    assert np.allclose(post_x_2, np.sum(post_grid_2, axis=(1, 2)) * dydz)
    assert np.allclose(post_y_1, np.sum(post_grid_1, axis=(0, 2)) * dxdz)
    assert np.allclose(post_y_2, np.sum(post_grid_2, axis=(0, 2)) * dxdz)
    assert np.allclose(post_z_1, np.sum(post_grid_1, axis=(0, 1)) * dxdy)
    assert np.allclose(post_z_2, np.sum(post_grid_2, axis=(0, 1)) * dxdy)

    # Convert 2D posteriors to confidence levels
    conf_xy_1 = post_to_conf(post_xy_1, dxdy)
    conf_xy_2 = post_to_conf(post_xy_2, dxdy)
    conf_xz_1 = post_to_conf(post_xz_1, dxdz)
    conf_xz_2 = post_to_conf(post_xz_2, dxdz)
    conf_yz_1 = post_to_conf(post_yz_1, dydz)
    conf_yz_2 = post_to_conf(post_yz_2, dydz)

    # Apply smoothing
    if smooth_sig_2d is not None:
        conf_xy_1 = scipy.ndimage.gaussian_filter(conf_xy_1, smooth_sig_2d)
        conf_xy_2 = scipy.ndimage.gaussian_filter(conf_xy_2, smooth_sig_2d)
        conf_xz_1 = scipy.ndimage.gaussian_filter(conf_xz_1, smooth_sig_2d)
        conf_xz_2 = scipy.ndimage.gaussian_filter(conf_xz_2, smooth_sig_2d)
        conf_yz_1 = scipy.ndimage.gaussian_filter(conf_yz_1, smooth_sig_2d)
        conf_yz_2 = scipy.ndimage.gaussian_filter(conf_yz_2, smooth_sig_2d)
    if smooth_sig_1d is not None:
        post_x_1 = scipy.ndimage.gaussian_filter(post_x_1, smooth_sig_1d)
        post_x_2 = scipy.ndimage.gaussian_filter(post_x_2, smooth_sig_1d)
        post_y_1 = scipy.ndimage.gaussian_filter(post_y_1, smooth_sig_1d)
        post_y_2 = scipy.ndimage.gaussian_filter(post_y_2, smooth_sig_1d)
        post_z_1 = scipy.ndimage.gaussian_filter(post_z_1, smooth_sig_1d)
        post_z_2 = scipy.ndimage.gaussian_filter(post_z_2, smooth_sig_1d)

    # Plot everything
    contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]
    dash = [(0, (5.0, 5.0))]
    plt.rcParams.update({'font.size': 13})
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex='col', figsize=(2 * plt.figaspect(1 / 1.4)))
    plt.subplots_adjust(wspace=0, hspace=0)

    # Row 0: x
    axes[0, 0].plot(x, post_x_1, color='c0', lw=3, label=like_label_1)
    axes[0, 0].plot(x, post_x_2, color='c1', ls=dash[0], lw=3, label=like_label_2)
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')

    # Row 1: x vs y, y
    axes[1, 0].contour(xy_x, xy_y, conf_xy_1, levels=contour_levels, colors='C0', linewidths=3)
    axes[1, 0].contour(xy_x, xy_y, conf_xy_2, levels=contour_levels, colors='C1', linewidths=3, linestyles=dash)
    axes[1, 1].plot(y, post_y_1, c='C0', lw=3)
    axes[1, 1].plot(y, post_y_2, c='C1', ls=dash[0], lw=3)
    axes[1, 2].axis('off')

    # Row 2: x vs z, y vs z, z
    axes[2, 0].contour(xz_x, xz_z, conf_xz_1, levels=contour_levels, colors='C0', linewidths=3)
    axes[2, 0].contour(xz_x, xz_z, conf_xz_2, levels=contour_levels, colors='C1', linewidths=3, linestyles=dash)
    axes[2, 1].contour(yz_y, yz_z, conf_yz_1, levels=contour_levels, colors='C0', linewidths=3)
    axes[2, 1].contour(yz_y, yz_z, conf_yz_2, levels=contour_levels, colors='C1', linewidths=3, linestyles=dash)
    axes[2, 2].plot(z, post_z_1, c='C0', lw=3)
    axes[2, 2].plot(z, post_z_2, c='C1', ls=dash[0], lw=3)

    # Hide y ticks for 1D posteriors
    axes[0, 0].tick_params(axis='y', which='both', left=False, labelleft=False)
    axes[1, 1].tick_params(axis='y', which='both', left=False, labelleft=False)
    axes[2, 2].tick_params(axis='y', which='both', left=False, labelleft=False)

    # Add x ticks at top and bottom of 2D posteriors and at bottom of 1D posteriors
    axes[0, 0].tick_params(axis='x', which='both', bottom=True, direction='in')
    axes[1, 0].tick_params(axis='x', which='both', top=True, bottom=True, direction='in')
    axes[2, 0].tick_params(axis='x', which='both', top=True, bottom=True, direction='inout', length=7.5)
    axes[0, 1].tick_params(axis='x', which='both', bottom=True, direction='in')
    axes[2, 1].tick_params(axis='x', which='both', top=True, bottom=True, direction='inout', length=7.5)
    axes[2, 2].tick_params(axis='x', which='both', bottom=True, direction='inout', length=7.5)

    # Add y ticks at left and right of 2D posteriors
    axes[1, 0].tick_params(axis='y', which='both', left=True, direction='inout', length=7.5)
    axes[1, 0].secondary_yaxis('right').tick_params(axis='y', which='both', right=True, direction='in',
                                                    labelright=False)
    axes[2, 0].tick_params(axis='y', which='both', left=True, right=True, direction='inout', length=7.5)
    axes[2, 1].tick_params(axis='y', which='both', left=True, right=True, labelleft=False, direction='in')

    # Label axes
    axes[2, 0].set_xlabel(x_label)
    axes[2, 1].set_xlabel(y_label)
    axes[2, 2].set_xlabel(z_label)
    axes[1, 0].set_ylabel(y_label)
    axes[2, 0].set_ylabel(z_label)
    fig.align_ylabels()

    # Legend
    leg_title = f'{min(contour_levels_sig)}\N{en dash}{max(contour_levels_sig)}$\\sigma$ confidence'
    axes[0, 0].legend(loc='upper right', bbox_to_anchor=(3, 1), handlelength=4, frameon=False, title=leg_title)

    # Limits
    axes[2, 0].set_xlim(x_lims)
    axes[2, 1].set_xlim(y_lims)
    axes[2, 2].set_xlim(z_lims)
    axes[1, 0].set_ylim(y_lims)
    axes[2, 0].set_ylim(z_lims)
    axes[2, 1].set_ylim(z_lims)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_2d(like_paths, labels, colours, linestyles, contour_levels_sig, smooth_sigma=0, x_label=None, y_label=None,
            x_lims=None, y_lims=None, true_params=None, fid_params=None, leg_title=None, annot=None, save_path=None):
    """
    Plot any number of 2D posteriors from any number of log-likelihood files containing any number of likelihood
    columns.

    Args:
        like_paths (list): List of log-likelihood files, e.g. ['like1.txt', 'like2.txt'].
        labels (list): Nested list of legend labels for each likelihood column in each file,
                       e.g. [['like1a', 'like1b'], ['like2']].
        colours (list): Line colours in the same nested list format as labels.
        linestyles (list): Line styles in the same nested list format as labels.
        contour_levels_sig (list): List of confidence intervals to plot, e.g. [1, 2, 3].
        smooth_sigma (float, optional): Size of smoothing kernel in standard deviations.
        x_label (str, optional): x-axis label.
        y_label (str, optional): y-axis label.
        x_lims ((float, float), optional): x-axis limits.
        y_lims ((float, float), optional): y-axis limits.
        true_params ((float, float), optional): Value of 'true' parameters to mark.
        fid_params ((float, float), optional): Value of fiducial parameters to mark.
        leg_title (str, optional): Legend title.
        annot (str, optional): Additional text under legend.
        save_path (str, optional): Path to save plot to, if supplied. If not supplied, plot is displayed.
    """

    # Load data and form various 'flat' lists (i.e. not nested)
    x_vals = []
    y_vals = []
    log_likes = []
    labels_flat = []
    colours_flat = []
    linestyles_flat = []
    for like_path, label, colour, linestyle in zip(like_paths, labels, colours, linestyles):

        # Determine how many columns to read by whether label is a single label or a list
        if isinstance(label, list):
            n_like_cols = len(label)
            labels_flat.extend(label)
            colours_flat.extend(colour)
            linestyles_flat.extend(linestyle)
        else:
            n_like_cols = 1
            labels_flat.append(label)
            colours_flat.append(colour)
            linestyles_flat.append(linestyle)

        # Load in the x and y columns and the appropriate number of log-likelihoods
        data = np.loadtxt(like_path)
        x = data[:, 0]
        y = data[:, 1]
        for col in range(n_like_cols):
            log_like = data[:, 2 + col]
            x_vals.append(x)
            y_vals.append(y)
            log_likes.append(log_like)

    # Beyond this point everything is a flat list, meaning no nested lists and no knowledge of which columns came from
    # which files

    # Convert log-likelihood to unnormalised posterior (flat prior) while aiming to prevent over/underflows
    posts = []
    for log_like in log_likes:
        posts.append(np.exp(log_like - np.amax(log_like) + 100))

    # Form x and y grids and determine grid cell size (requires and checks for regular grid)
    dxdys = []
    x_grids = []
    y_grids = []
    for x_val, y_val in zip(x_vals, y_vals):
        x_val_unique = np.unique(x_val)
        dx = x_val_unique[1] - x_val_unique[0]
        assert np.allclose(np.diff(x_val_unique), dx)
        y_val_unique = np.unique(y_val)
        dy = y_val_unique[1] - y_val_unique[0]
        assert np.allclose(np.diff(y_val_unique), dy)
        dxdys.append(dx * dy)

        x_grid, y_grid = np.meshgrid(x_val_unique, y_val_unique)
        x_grids.append(x_grid)
        y_grids.append(y_grid)

    # Grid and convert to confidence intervals
    conf_grids = []
    for x_val, y_val, post, x_grid, y_grid, dxdy in zip(x_vals, y_vals, posts, x_grids, y_grids, dxdys):
        post_grid = scipy.interpolate.griddata((x_val, y_val), post, (x_grid, y_grid), fill_value=0)
        if contour_levels_sig is not None:
            conf_grids.append(post_to_conf(post_grid, dxdy))
        else:
            conf_grids.append(post_grid) # Plot raw posterior

    # Calculate contour levels
    if contour_levels_sig is not None:
        contour_levels = [0.] + [scipy.special.erf(contour_level / np.sqrt(2)) for contour_level in contour_levels_sig]
    else:
        contour_levels = None

    # Smoothing
    conf_grids = [scipy.ndimage.gaussian_filter(conf_grid, smooth_sigma) for conf_grid in conf_grids]

    # Plot everything
    plt.rcParams.update({'font.size': 13})
    contours = []
    for x_grid, y_grid, conf_grid, colour, ls in zip(x_grids, y_grids, conf_grids, colours_flat, linestyles_flat):
        cont = plt.contour(x_grid, y_grid, conf_grid, levels=contour_levels, colors=colour, linestyles=ls,
                           linewidths=3)
        contours.append(cont)
        # plt.scatter(x_grid.flatten(), y_grid.flatten(), s=2) # uncomment to show grid points

    # True input parameters
    if true_params is not None:
        plt.scatter(*true_params, marker='+', c='navy', s=90, label='True input parameters')

    # Fiducial cosmology
    if fid_params is not None:
        plt.scatter(*fid_params, marker='x', c='navy', s=90, label='Fiducial parameters')

    # Legend
    handles_init, labels_init = plt.gca().get_legend_handles_labels()
    if contour_levels_sig is None:
        arb_str = '(arbitrary contours)'
        leg_title = leg_title + ' ' + arb_str if leg_title is not None else arb_str
        sig_label = None
    else:
        sig_label = f'{min(contour_levels_sig)}\N{en dash}{max(contour_levels_sig)}$\\sigma$ confidence'
        leg_title = f'{leg_title} ({sig_label})' if leg_title is not None else sig_label
    handles_final = []
    labels_final = []
    for cont, label in zip(contours, labels_flat):
        h, _ = cont.legend_elements()
        handles_final.append(h[0])
        labels_final.append(label + (f' {sig_label}' if contour_levels_sig and not leg_title else ''))
    handles_final.extend(handles_init)
    labels_final.extend(labels_init)
    plt.legend(handles_final, labels_final, title=leg_title, frameon=False, handlelength=4)

    # Limits and axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_lims)
    plt.ylim(y_lims)

    # Arbitrary annotation below legend (coordinates semi-hardcoded)
    n_leg_rows = len(labels_flat)
    n_leg_rows += 2 if leg_title else 1
    annot_y = 0.85 - n_leg_rows * 0.07
    if annot is not None:
        plt.annotate(annot, xy=(0.97, annot_y), xycoords='axes fraction', ha='right', va='top')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_post_max(input_filemask, save_path=None):
    """
    Plot histogram of Wishart and Gaussian 1D posterior maxima and per-realisation difference,
    as calculated in run_likelihoods.max_like_1d.

    Args:
        input_filemask (str): glob filemask of maximum likelihood parameters output by run_likelihoods.max_like_1d.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure is displayed.
    """

    # Load and combine input
    data = np.concatenate([np.loadtxt(f) for f in glob.glob(input_filemask)])
    postmax_w = data[:, 0]
    postmax_g = data[:, 1]

    # Plot hists using bins that respect the inherent discretisation of the data,
    # following https://stackoverflow.com/a/30121210
    step = np.amin(np.diff(np.unique(data)))
    hist_min = np.amin(data) - 0.5 * step
    hist_max = np.amax(data) + 0.5 * step
    bins = np.arange(hist_min, hist_max, step)
    diff = postmax_g - postmax_w
    diff_min = np.amin(diff) - 0.5 * step
    diff_max = np.amax(diff) + 0.5 * step
    diff_bins = np.arange(diff_min, diff_max, step)

    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(ncols=2, figsize=plt.figaspect(1 / 3.))
    plt.subplots_adjust(wspace=.3)

    ax[0].hist(postmax_w, bins=bins, histtype='step', lw=3, label='Wishart')
    ax[0].hist(postmax_g, bins=bins, histtype='step', lw=3, ls=(0, (5, 3)), label='Gaussian')
    ax[0].axvline(x=-1, c='grey', ls='--', label='Input')
    ax[0].legend(loc='upper left', frameon=False)
    ax[0].set_xlabel(r'Maximum posterior $w_0$')
    ax[0].set_ylabel('Number of realisations')

    ax[1].hist(diff, bins=diff_bins, weights=(np.ones_like(diff) / len(diff)), color='grey', ec='k')
    ax[1].set_xlabel('Difference in posterior maximum')
    ax[1].set_ylabel('Fraction of realisations')
    ax[1].set_yscale('log')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_post_mean_std(input_filemask, save_path=None):
    """
    Plot histograms of Wishart and Gaussian posterior mean and standard deviation distributions,
    as produced by run_likelihoods.post_mean_std_1d.

    Args:
        input_filemask (str): glob filemask of posterior mean and standard deviation output by
                              run_likelihoods.max_like_1d.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure is displayed.
    """

    # Load and combine input
    data = np.concatenate([np.loadtxt(f) for f in glob.glob(input_filemask)])
    mean_w = data[:, 0]
    mean_g = data[:, 1]
    std_w = data[:, 2]
    std_g = data[:, 3]

    # Bin the data - not naturally discretised here in the same way as postmax
    nbin = 50
    mean_bins = np.linspace(np.amin((mean_w, mean_g)), np.amax((mean_w, mean_g)), nbin)
    std_bins = np.linspace(np.amin((std_w, std_g)), np.amax((std_w, std_g)), nbin)

    # Plot side by side
    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(ncols=2, figsize=plt.figaspect(1 / 3.))
    plt.subplots_adjust(wspace=.3)

    ax[0].hist(mean_w, bins=mean_bins, histtype='step', lw=3, label='Wishart')
    ax[0].hist(mean_g, bins=mean_bins, histtype='step', lw=3, ls=(0, (5, 3)), label='Gaussian')
    ax[0].axvline(x=-1, c='grey', ls='--', label='Input')
    ax[0].legend(loc='upper left', frameon=False)
    ax[0].set_xlabel(r'Posterior mean $w_0$')
    ax[0].set_ylabel('Number of realisations')

    ax[1].hist(std_w, bins=std_bins, histtype='step', lw=3, label='Wishart')
    ax[1].hist(std_g, bins=std_bins, histtype='step', lw=3, ls=(0, (5, 3)), label='Gaussian')
    ax[1].legend(frameon=False)
    ax[1].set_xlabel(r'Posterior standard deviation $\sigma (w_0)$')
    ax[1].set_ylabel('Number of realisations')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()
