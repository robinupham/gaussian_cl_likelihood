"""
Contains functions relating to skewness and kurtosis.
"""

import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.special


def skew_fullsky(l):
    """
    Returns the skewness of the full-sky marginal likelihood for a given l.

    Args:
        l (int): The l to return the skewness for.
    Returns:
        float: Skewness of the full-sky marginal likelihood for this l.
    """
    nu = 2 * l + 1
    k = nu / 2.
    return 2 / np.sqrt(k)


def exkurt_fullsky(l):
    """
    Returns the excess kurtosis of the full-sky marginal likelihood for a given l.

    Args:
        l (int): The l to return the excess kurtosis for.
    Returns:
        float: Excess kurtosis of the full-sky marginal likelihood for this l.
    """
    nu = 2 * l + 1
    k = nu / 2.
    return 6 / k


def ex_std_moment(x, pdf, order):
    """
    Numerically calculate the excess standard moment of a distribution for a given order,
    i.e. for order n, <((x - μ)/σ)^n> minus the Gaussian equivalent, where μ is the mean and σ the standard deviation.

    Args:
        x (1D numpy array): Values of the variable whose moments are required.
        pdf (1D numpy array): Normalised probability density values at each value of x.
        order (int): Order of the required moment, e.g. 3 for skewness.

    Returns:
        float: Excess standard moment of the required order.
    """

    # Work out the mean and standard deviation and use them to calculate standardised x (= (x - μ)/σ)
    dx = x[1] - x[0]
    mean = np.sum(x * pdf) * dx
    var = np.sum((x - mean) ** 2 * pdf) * dx
    x_std = (x - mean) / np.sqrt(var)

    # Calculate the moment and subtract the Gaussian value
    mom = np.sum(x_std ** order * pdf) * dx
    mom_gauss = 0 if order % 2 else scipy.special.factorial2(order - 1, exact=True)
    return mom - mom_gauss


def get_all_moments(cut_sky_pdf_path, lmax, plot=None, plot_save_path=None, data_save_path=None):
    """
    Calculate full-sky and cut-sky skewness and excess kurtosis for all l.

    Args:
        cut_sky_pdf_path (str): Path to marginal pseudo-Cl likelihood distributions
                                produced by pcl_like.marginal_cl_likes.
        lmax (int): Maximum l to calculate moments for.
        plot (str, optional): 'skew' to plot skewness, 'exkurt' to plot excess kurtosis.
        plot_save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure is displayed.
        data_save_path (str, optional): Path to save calculated moments to as a text file, if supplied.
    """

    # Load cut-sky pdfs
    with np.load(cut_sky_pdf_path) as data:
        lmin = data['lmin']
        pdfs_ma = data['pdfs']

    # Calculate full-sky and cut-sky skewness and excess kurtosis for all l
    skew_fsan = []
    kurt_fsan = []
    skew_ma = []
    kurt_ma = []
    for l, (x, pdf_ma) in enumerate(pdfs_ma[:(lmax - lmin + 1)], start=lmin):
        print(f'l = {l}', end='\r')

        # Skewness
        skew_fsan.append(skew_fullsky(l))
        skew_ma.append(ex_std_moment(x, pdf_ma, 3))

        # Excess kurtosis
        kurt_fsan.append(exkurt_fullsky(l))
        kurt_ma.append(ex_std_moment(x, pdf_ma, 4))

    ell = range(lmin, lmin + len(skew_ma))

    # Plot skewness or excess kurtosis
    if plot == 'skew':
        plt.plot(ell, skew_fsan, label='Full sky')
        plt.plot(ell, skew_ma, label='Cut sky')
        plt.ylabel('Skewness')
    elif plot == 'exkurt':
        plt.plot(ell, kurt_fsan, label='Full sky')
        plt.plot(ell, kurt_ma, label='Cut sky')
        plt.ylabel('Excess kurtosis')
    if plot is not None:
        plt.plot(ell, np.zeros_like(ell), ls='--', label='Gaussian')
        plt.xlabel(r'$\ell$')
        plt.legend()
        if plot_save_path is not None:
            plt.savefig(plot_save_path, bbox_inches='tight')
            print('Saved ' + plot_save_path)
        else:
            plt.show()

    # Save data to file
    if data_save_path is not None:
        data = np.column_stack((ell, skew_fsan, skew_ma, kurt_fsan, kurt_ma))
        header = 'ell skewness_fs skewness_ma ex_kurtosis_fs ex_kurtosis_ma'
        np.savetxt(data_save_path, data, header=header)
        print('Saved ' + data_save_path)


def explain_mapping(cut_sky_pdf_path, lmax, save_path=None):
    """
    Produce a pictorial aid for the l-mapping process using skewness.

    Args:
        cut_sky_pdf_path (str): Path to marginal pseudo-Cl likelihood distributions
                                produced by pcl_like.marginal_cl_likes.
        lmax (int): Maximum l to calculate skewness for.
        plot_save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure is displayed.
    """

    # Load cut-sky pdfs
    with np.load(cut_sky_pdf_path) as data:
        lmin = data['lmin']
        pdfs_ma = data['pdfs']

    # Calculate skewness for all l
    skew_fsan = []
    skew_ma = []
    for l, (x, pdf_ma) in enumerate(pdfs_ma[:(lmax - lmin + 1)], start=lmin):
        skew_fsan.append(skew_fullsky(l))
        skew_ma.append(ex_std_moment(x, pdf_ma, 3))

    # Plot skewness against l
    ell = range(lmin, lmin + len(skew_ma))
    plt.rcParams.update({'font.size': 13})
    plt.plot(ell, skew_fsan, lw=2.5, c='C0', label='Full sky')
    plt.plot(ell, skew_ma, lw=2.5, c='C1', label='Cut sky')
    plt.plot(ell, np.zeros_like(ell), ls='--', lw=2.5, c='C2', label='Gaussian')
    plt.xlabel(r'$\ell$')
    plt.ylabel('Skewness')

    # Add lines to explain mapping
    map_ells = [20, 45, 75]
    skew_fsan_neg = -1 * np.array(skew_fsan)
    for map_l in map_ells:
        l_idx = ell.index(map_l)
        skew_ma_l = skew_ma[l_idx]
        leff = np.interp(-skew_ma_l, skew_fsan_neg, ell)
        plt.vlines(map_l, -1, skew_ma_l, ls='dotted', lw=2, colors='k')
        plt.arrow(map_l, skew_ma_l, (leff - map_l), 0, length_includes_head=True, head_width=.2, head_length=3,
                  overhang=1, lw=2.5, zorder=10)
        plt.vlines(leff, -1, skew_ma_l, ls='dotted', lw=2, colors='k')

    # Hack to get arrow in legend
    arrow_label = r'$\ell \rightarrow \ell_\mathrm{eff}$ mapping'
    plt.scatter(-5, -5, c='k', marker=r'$\longleftarrow$', s=1000, label=arrow_label)
    plt.legend(frameon=False, handlelength=4)

    plt.xlim(1, 80)
    plt.ylim(-0.1, max(skew_ma))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def get_leff_map(data_path, fit_lmax, lmax, grid_vals=None, plot_save_path=None, map_save_path_kurt=None,
                 map_save_path_skew=None):
    """
    Calculate l-leff mapping from full-sky and cut-sky moments calculated in get_all_moments.

    Args:
        data_path (str): Path to moments produced by get_all_moments.
        fit_lmax (int): Maximum l to use for the linear fit used for the extrapolation.
        lmax (int): Maximum l to extrapolate to.
        grid_vals (list, optional): List of x values at which to draw grid lines, which extend from the x axis to the
                                    mapping line and then left to the y axis.
        plot_save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure is displayed.
        map_save_path_kurt (str, optional): Path to save l-leff mapping calculated using kurtosis.
        map_save_path_skew (str, optional): Path to save l-leff mapping calculated using skewness.
    """

    # Load data
    ell, skew_fs, skew_ma, kurt_fs, kurt_ma = np.loadtxt(data_path, unpack=True)

    # Map by linear interpolation (negative so that the x range is increasing)
    leff_skew = np.interp(-skew_ma, -skew_fs, ell, left=np.NINF, right=np.inf)
    leff_kurt = np.interp(-kurt_ma, -kurt_fs, ell, left=np.NINF, right=np.inf)
    plt.plot(ell, leff_skew, label='Skewness')
    plt.plot(ell, leff_kurt, label='Kurtosis')

    # Fit a line over the finite range up to fit_lmax
    keep_skew = np.logical_and(np.isfinite(leff_skew), ell <= fit_lmax)
    grad_skew, int_skew, _, _, _ = scipy.stats.linregress(ell[keep_skew], leff_skew[keep_skew])
    keep_kurt = np.logical_and(np.isfinite(leff_kurt), ell <= fit_lmax)
    grad_kurt, int_kurt, _, _, _ = scipy.stats.linregress(ell[keep_kurt], leff_kurt[keep_kurt])

    # Use the line to extrapolate up to lmax
    lmin = min(ell)
    ell_full = np.arange(lmin, lmax + 1)
    lin_skew = grad_skew * ell_full + int_skew
    lin_kurt = grad_kurt * ell_full + int_kurt
    plt.plot(ell_full, lin_skew, ls='--', c='C0')
    plt.plot(ell_full, lin_kurt, ls='--', c='C1')

    # Add grid lines
    if grid_vals is not None:
        grid_x = np.array(grid_vals)
        grid_y = grad_kurt * grid_x + int_kurt
        plt.vlines(grid_x, np.zeros_like(grid_x), grid_y, colors='grey', alpha=0.5, lw=.5)
        plt.hlines(grid_y, np.zeros_like(grid_y), grid_x, colors='grey', alpha=0.5, lw=.5)

    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell_\mathrm{eff}$')
    plt.legend(loc='upper left', ncol=3, frameon=False)

    if plot_save_path is not None:
        plt.savefig(plot_save_path, bbox_inches='tight')
        print('Saved ' + plot_save_path)
    else:
        plt.show()

    # Save mapping to file
    if map_save_path_kurt is not None:
        data = np.column_stack((ell_full, lin_kurt))
        header = f'Kurtosis mapping from {__file__} for input {data_path} at {time.strftime("%c")} \n'
        header += 'l l_eff'
        np.savetxt(map_save_path_kurt, data, header=header)
        print('Saved ' + map_save_path_kurt)
    if map_save_path_skew is not None:
        data = np.column_stack((ell_full, lin_skew))
        header = f'Skewness mapping from {__file__} for input {data_path} at {time.strftime("%c")} \n'
        header += 'l l_eff'
        np.savetxt(map_save_path_skew, data, header=header)
        print('Saved ' + map_save_path_skew)


def plot_leff_map(data_path, fit_lmax, lmax, plot_save_path=None):
    """
    Plot exact and extrapolated l-leff mapping side-by-side.

    Args:
        data_path (str): Path to moments produced by get_all_moments.
        fit_lmax (int): Maximum l to use for the linear fit used for the extrapolation.
        lmax (int): Maximum l to extrapolate to.
        plot_save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure is displayed.
    """

    # Load data and cut at lmax
    ell, skew_fs, skew_ma, kurt_fs, kurt_ma = np.loadtxt(data_path, unpack=True)

    # Map by linear interpolation (negative so that the x range is increasing)
    leff_skew = np.interp(-skew_ma, -skew_fs, ell, left=np.NINF, right=np.inf)
    leff_kurt = np.interp(-kurt_ma, -kurt_fs, ell, left=np.NINF, right=np.inf)

    # Fit a line over the finite range up to fit_lmax
    keep_skew = np.logical_and(np.isfinite(leff_skew), ell <= fit_lmax)
    grad_skew, int_skew, _, _, _ = scipy.stats.linregress(ell[keep_skew], leff_skew[keep_skew])
    keep_kurt = np.logical_and(np.isfinite(leff_kurt), ell <= fit_lmax)
    grad_kurt, int_kurt, _, _, _ = scipy.stats.linregress(ell[keep_kurt], leff_kurt[keep_kurt])

    # Use the line to extrapolate up to lmax
    lmin = min(ell)
    ell_full = np.arange(lmin, lmax + 1)
    lin_skew = grad_skew * ell_full + int_skew
    lin_kurt = grad_kurt * ell_full + int_kurt

    # Plot lmax 80 and lmax 2000 side by side
    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(ncols=2, figsize=plt.figaspect(1 / 3.))
    plt.subplots_adjust(wspace=.3)

    ax[0].plot(ell[keep_skew], leff_skew[keep_skew], lw=2.5, c='C0', label='Skewness')
    ax[0].plot(ell[keep_kurt], leff_kurt[keep_kurt], lw=2.5, c='C1', label='Kurtosis')
    ax[1].plot(ell_full, lin_skew, ls=(0, (3, 1)), lw=2.5, c='C0', label='Skewness')
    ax[1].plot(ell_full, lin_kurt, ls=(2, (3, 1)), lw=2.5, c='C1', label='Kurtosis')
    ax[1].plot(ell[keep_skew], leff_skew[keep_skew], lw=2.5, c='C0')
    ax[1].plot(ell[keep_kurt], leff_kurt[keep_kurt], lw=2.5, c='C1')

    # Add grid lines
    grid_vals_low = np.arange(20, 81, 20)
    grid_vals_full = np.arange(250, 2001, 250)
    for a, grid_x in zip(ax, [grid_vals_low, grid_vals_full]):
        grid_y = grad_kurt * grid_x + int_kurt
        a.vlines(grid_x, np.zeros_like(grid_x), grid_y, colors='grey', alpha=0.5, lw=.5)
        a.hlines(grid_y, np.zeros_like(grid_y), grid_x, colors='grey', alpha=0.5, lw=.5)
        a.set_xlim(left=0)
        a.set_ylim(bottom=0)

    # Label panels
    panels = ['Exact', 'Extrapolated']
    for a, annot in zip(ax, panels):
        a.set_xlabel(r'$\ell$')
        a.set_ylabel(r'$\ell_\mathrm{eff}$')
        a.legend(loc='upper left', bbox_to_anchor=(0, 0.92), ncol=2, frameon=False)
        a.annotate(annot, xy=(0.5, 0.935), ha='center', xycoords='axes fraction')

    if plot_save_path is not None:
        plt.savefig(plot_save_path, bbox_inches='tight')
        print('Saved ' + plot_save_path)
    else:
        plt.show()


def get_sim_moments(input_files, data_id, batch_size, n_ell, whiten=False):
    """
    Extract sample skewness and excess kurtosis from a list of input files produced by simulation.sim_singlespec.

    Args:
        input_files (list): List of input files to combine before measuring skewness and excess kurtosis.
        data_id (str): Data label within input npz files.
        batch_size (int): Number of realisations per batch.
        n_ell (int): Number of ells.
        whiten (bool, optional): If True, whiten the data before measuring moments (default False).

    Returns:
        (1D numpy array, 1D numpy array): Skewness for each l, excess kurtosis for each l.
    """

    n_input = len(input_files)
    cl = np.full((n_input * batch_size, n_ell), np.nan)
    for i, input_file in enumerate(input_files):
        print(f'Loading {i + 1} / {n_input}')
        with np.load(input_file) as input_data:
            cl[i * batch_size:(i + 1) * batch_size] = input_data[data_id]
    assert np.all(np.isfinite(cl))

    if whiten:
        w = np.linalg.cholesky(np.linalg.inv(np.cov(cl, rowvar=False))).T
        cl = (w @ cl.T).T
        cl -= np.mean(cl, axis=0)
        assert np.allclose(np.cov(cl, rowvar=False), np.identity(n_ell))
        assert np.allclose(np.mean(cl, axis=0), 0)

    print('Calculating skewness')
    skew = scipy.stats.skew(cl, axis=0, bias=False)
    print('Calculating excess kurtosis')
    kurt = scipy.stats.kurtosis(cl, axis=0, fisher=True, bias=False)

    return skew, kurt


def save_sim_moments(input_filemask, batch_size, lmax, lmin, save_path, whiten=False):
    """
    Calculate full-sky and cut-sky skewness and kurtosis for all l from simulations produced with
    simulation.sim_singlespec, and save the results to a text file.

    Args:
        input_filemask (str): glob filemask matching all simulation batches produced with simulation.sim_singlespec.
        batch_size (int): Number of realisations per batch.
        lmax (int): Maximum l of simulations.
        lmin (int): Minimum l of simulations.
        save_path (str): Path to save output text file to.
        whiten (bool, optional): If True, whiten data prior to measuring moments (default False).
    """

    # Calculate skewness and excess kurtosis for each l for all files combined
    input_files = glob.glob(input_filemask)
    skew_fs, kurt_fs = get_sim_moments(input_files, 'cl_fs', batch_size, lmax - lmin + 1, whiten=whiten)
    skew_ma, kurt_ma = get_sim_moments(input_files, 'cl_ma', batch_size, lmax - lmin + 1, whiten=whiten)

    # Save to a text file
    ell = np.arange(lmin, lmax + 1)
    output = np.column_stack((ell, skew_fs, skew_ma, kurt_fs, kurt_ma))
    header = (f'Output from {__file__}.save_sim_moments for {len(input_files)} input files with batch size {batch_size}'
              f' at {time.strftime("%c")}\n')
    header += 'ell skew_fs skew_ma exkurt_fs exkurt_ma'
    np.savetxt(save_path, output, header=header)
    print('Saved ' + save_path)


def plot_sim_moments(sim_path, kurt_leff_map_path, skew_leff_map_path, save_path=None):
    """
    Plot simulated moments calculated with save_sim_moments compared to moments predicted by l-leff mapping produced
    with get_leff_map.

    Args:
        sim_path (str): Path to simulated moments produced by save_sim_moments.
        kurt_leff_map_path (str): Path to kurtosis l-leff mapping produced by get_leff_map.
        skew_leff_map_path (str): Path to skewness l-leff mapping produced by get_leff_map.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure is displayed.
    """

    # Load simulated data
    ell, _, skew_ma, _, kurt_ma = np.loadtxt(sim_path, unpack=True)

    # Load leff maps and calculate cut-sky gamma parameters
    leff_kurt = np.loadtxt(kurt_leff_map_path)[:, 1]
    leff_skew = np.loadtxt(skew_leff_map_path)[:, 1]
    k_eff_kurt = (2 * leff_kurt + 1) / 2
    k_eff_skew = (2 * leff_skew + 1) / 2
    skew_eff = 2 / np.sqrt(k_eff_skew)
    kurt_eff = 6 / k_eff_kurt

    # Plot with low-l cut
    skew_lmin = 60
    kurt_lmin = 20
    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(ncols=2, figsize=plt.figaspect(1 / 3.))
    plt.subplots_adjust(wspace=.3)

    # Skewness
    ax[0].plot(ell[ell >= skew_lmin], skew_ma[ell >= skew_lmin], label='Simulations')
    ax[0].plot(ell[ell >= skew_lmin], skew_eff[ell >= skew_lmin], ls='--', lw=2.5, label=('Effective'))
    ax[0].set_ylabel('Skewness')

    # Kurtosis
    ax[1].plot(ell[ell >= kurt_lmin], kurt_ma[ell >= kurt_lmin], label='Simulations')
    ax[1].plot(ell[ell >= kurt_lmin], kurt_eff[ell >= kurt_lmin], ls='--', lw=2.5, label='Effective')
    ax[1].set_ylabel('Excess kurtosis')

    for a in ax:
        a.set_xlabel(r'$\ell$')
        a.legend(loc='upper right', frameon=False, handlelength=4)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_skew_slics_gauss(slics_filemask, gauss_path, n_zbin, save_path=None):
    """
    Plot skewness against l for SLICS compared to Gaussian field simulations.

    Args:
        slics_filemask (str): Path to SLICS bandpowers output by simulation.combine_slics,
                              with {tomo} as a placeholder for tomographic bin.
        gauss_path (str): Path to Gaussian field bandpowers output by simulation.sim_flat.
        n_zbin (int): Number of redshift bins.
        save_path (str, optional): Path to save figure to, if supplied. If not supplied, figure is displayed.
    """

    # Import Gaussian field data
    with np.load(gauss_path) as data:
        leff = data['leff']
        bp_gauss = data['bps']
    n_bp = len(leff)
    n_real = bp_gauss.shape[0]
    n_spec = n_zbin * (n_zbin + 1) // 2
    assert bp_gauss.shape[1:] == (n_spec, n_bp)

    # Import SLICS data
    bp_slics = np.full((n_real, n_zbin, n_bp), np.nan)
    for tomo in range(n_zbin):
        with np.load(slics_filemask.format(tomo=tomo)) as data:
            assert data['tomo'] == tomo
            assert np.allclose(data['l'], leff)
            bp_slics[:, tomo, :] = data['cl']
    assert np.all(np.isfinite(bp_slics))

    # Extract only the auto-spectra from the Gaussian sims, where they are stored in row-major order (NaMaster-style)
    auto_indices = np.concatenate(([0], np.cumsum(np.arange(n_zbin, 1, -1))))
    bp_gauss = bp_gauss[:, auto_indices, :]
    assert bp_gauss.shape == bp_slics.shape

    # Calculate skewness against l for all tomos combined
    skew_gauss = []
    skew_slics = []
    for tomo in range(1, n_zbin): # exclude tomo0

        tomo_gauss = bp_gauss[:, tomo, :]
        tomo_slics = bp_slics[:, tomo, :]

        # Whiten the data
        w_slics = np.linalg.cholesky(np.linalg.inv(np.cov(tomo_slics, rowvar=False))).T
        w_gauss = np.linalg.cholesky(np.linalg.inv(np.cov(tomo_gauss, rowvar=False))).T
        tomo_slics = (w_slics @ tomo_slics.T).T
        tomo_gauss = (w_gauss @ tomo_gauss.T).T
        tomo_slics -= np.mean(tomo_slics, axis=0)
        tomo_gauss -= np.mean(tomo_gauss, axis=0)
        assert np.allclose(np.cov(tomo_slics, rowvar=False), np.identity(n_bp))
        assert np.allclose(np.cov(tomo_gauss, rowvar=False), np.identity(n_bp))
        assert np.allclose(np.mean(tomo_slics, axis=0), 0)
        assert np.allclose(np.mean(tomo_gauss, axis=0), 0)

        # Skewness
        skew_gauss.append(scipy.stats.skew(tomo_gauss, axis=0, bias=False))
        skew_slics.append(scipy.stats.skew(tomo_slics, axis=0, bias=False))

    # Plot mean skewness against l
    skew_gauss = np.array(skew_gauss)
    skew_slics = np.array(skew_slics)
    mean_skew_gauss = np.mean(skew_gauss, axis=0)
    mean_skew_slics = np.mean(skew_slics, axis=0)
    plt.rcParams.update({'font.size': 13})
    plt.plot(mean_skew_gauss, c='C0', lw=3, alpha=.8, label='Gaussian fields')
    plt.plot(mean_skew_slics, c='C1', lw=3, alpha=.8, label='SLICS')
    plt.xlabel('Whitened bandpower')
    plt.ylabel('Skewness')
    plt.legend(frameon=False, handlelength=4)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()
