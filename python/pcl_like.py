"""
Contains functions relating to the exact pseudo-Cl likelihood.
"""

import time

import numpy as np

import pseudo_cl_likelihood.alm_cov # https://github.com/robinupham/pseudo_cl_likelihood
import pseudo_cl_likelihood.cf_to_pdf


def marginal_alm_cov(cl_path, w0_path, lmax_in, lmin_in, lmax_out, lmin_out, save_path):
    """
    Calculate and save the marginal pseudo-alm covariance for each l from lmin_out to lmax_out, including mixing to/from
    all l from lmin_in to lmax_in.

    Args:
        cl_path (str): Path to theory power spectrum.
        w0_path (str): Path to W0 object output by pseudo_cl_likelihood.mask_to_w.combine_w_files.
        lmax_in (int): Maximum l to include mixing to/from.
        lmin_in (int): Minimum l to include mixing to/from.
        lmax_out (int): Maximum l to save marginal pseudo-alm covariance for.
        lmin_out (int): Minimum l to save marginal pseudo-alm covariance for.
        save_path (str): Path to save all covariances to.
    """

    # Load theory Cls
    theory_cl = pseudo_cl_likelihood.alm_cov.load_theory_cls(cl_path, 2, lmax_in, lmin_in)

    # Setup for a single spin-0 spectrum
    spins = [0]
    w_paths = [w0_path]
    theory_cls = [(0, '0', 0, '0', theory_cl)]

    # Calculate marginal alm cov for each l in turn
    covs = []
    for l_out in range(lmin_out, lmax_out + 1):
        print(f'Calculating l = {l_out} / {lmax_out}')
        cov = pseudo_cl_likelihood.alm_cov.pseudo_alm_cov(spins, w_paths, theory_cls, lmax_in, lmin_in, l_out, l_out)
        covs.append(cov)

    header = (f'Output from {__file__}.marginal_alm_cov for cl_path = {cl_path}, w0_path = {w0_path} '
              f'at {time.strftime("%c")}')
    np.savez_compressed(save_path, covs=covs, lmin_in=lmin_in, lmax_in=lmax_in, lmin_out=lmin_out, lmax_out=lmax_out,
                        header=header)
    print('Saved ' + save_path)


def marginal_cl_like(l, alm_cov, steps=100000):
    """
    Returns the exact cut-sky marginal likelihood distribution for a single l, given a pseudo-alm covariance matrix.

    Args:
        l (int): The l to calculate the marginal likelihood for.
        alm_cov (2D numpy array): Pseudo-alm covariance for this l.
        steps (int, optional): Resolution of the characteristic function, which translates to range in the
                               likelihood (default 100000).
    """

    # Form selection matrix and calculate eigenvalues of M @ cov
    m = np.diag(np.concatenate(([1], 2 * np.ones(2 * l)))) / (2 * l + 1.)
    evals = np.linalg.eigvals(m @ alm_cov)

    # Calculate FFT parameters via characteristic scale
    scale = np.sum(np.abs(evals))
    tmax = 400. / scale
    t = np.linspace(-tmax, tmax, steps - 1)

    # Calculate CF(t)
    cf = np.ones_like(t)
    for eigenval in evals:
        cf = np.multiply(cf, np.power(1 - 2j * eigenval * t, -0.5))

    # Calculate pdf from cf
    dt = t[1] - t[0]
    x, pdf = pseudo_cl_likelihood.cf_to_pdf.cf_to_pdf(cf=cf, t0=-tmax, dt=dt)

    # Avoid boundary issues by cutting to the range 0 -> 10 * scale
    x_cut = x[(x >= 0) & (x <= 10 * scale)]
    pdf_cut = pdf[(x >= 0) & (x <= 10 * scale)]
    return x_cut, pdf_cut


def marginal_cl_likes(covs_path, save_path):
    """
    Wrapper for marginal_cl_like to generate and save the marginal pseudo-Cl likelihood for each l, using the pseudo-alm
    covariances output by marginal_alm_cov.

    Args:
        covs_path (str): Path to pseudo-alm covariance matrices as output by marginal_alm_cov.
        save_path (str): Path to save output.
    """

    # Load pseudo-alm covariance matrices
    with np.load(covs_path, allow_pickle=True) as data:
        lmin = data['lmin_out']
        lmax = data['lmax_out']
        covs = data['covs']

    # Calculate exact marginal pdf for each l
    res = []
    for l, cov in zip(range(lmin, lmax + 1), covs):
        print(f'l = {l} / {lmax}', end='\r')
        res.append(np.stack(marginal_cl_like(l, cov)))

    # Save the result
    pdfs = np.array(res)
    header = (f'Marginal exact cut-sky pdfs. First axis is indexed [l - lmin], '
              f'second axis is 0 for x values (Cls) and 1 for pdf. Output from {__file__}.marginal_cl_likes for input '
              f'covs_path = {covs_path}, time {time.strftime("%c")}')
    np.savez_compressed(save_path, pdfs=pdfs, lmin=lmin, header=header)
    print('Saved ' + save_path)
