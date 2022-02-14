"""
Contains functions to run likelihood modules.
"""

import glob
import os.path
import time
import warnings

import numpy as np

from . import like_cl_gauss as like_g
from . import like_cl_wishart as like_w
from . import posteriors


def run_like_cl_wishart(grid_dir, varied_params, save_path, n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir,
                        pos_nl_path, she_nl_path, noise_ell_path, lmax, leff_path=None):
    """
    Evaluate Wishart likelhood on a CosmoSIS output grid and save the result as a text file.

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        varied_params (list): List of varied parameter names as they appear in the cosmological_parameters/values.txt
                              file.
        save_path (str): Path to save output to.
        n_zbin (int): Number of redshift bins. It will be assumed that there is one position field and one shear field
                      per redshift bin.
        obs_pos_pos_dir (str): Path to the directory containing the observed position-position power spectra.
        obs_she_she_dir (str): Path to the directory containing the observed shear-shear power spectra.
        obs_pos_she_dir (str): Path to the directory containing the observed position-shear power spectra.
        pos_nl_path (str): Path to the position noise power spectrum.
        she_nl_path (str): Path to the shear noise power spectrum.
        noise_ell_path (str): Path to the file containing the ells for the noise power spectra.
        lmax (int): Maximum l to use in the likelihood.
        leff_path (str, optional): Path to ell-ell_effective mapping, to replace each l with its corresponding l_eff
                                   when calculating the covariance.
    """

    print(f'Starting at {time.strftime("%c")}', flush=True)

    # Setup the likelihood module
    print(f'Setting up likelihood module at {time.strftime("%c")}')
    config_w = like_w.setup(n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir, pos_nl_path, she_nl_path,
                            noise_ell_path, lmax, leff_path=leff_path)
    print(f'Setup complete at {time.strftime("%c")}')

    # Loop over every input directory
    source_dirs = glob.glob(os.path.join(grid_dir, '_[0-9]*/'))
    n_dirs = len(source_dirs)
    if n_dirs == 0:
        warnings.warn(f'No matching directories. Terminating at {time.strftime("%c")}')
        return
    n_params = len(varied_params)
    if n_params == 0:
        warnings.warn(f'No parameters specified. Terminating at {time.strftime("%c")}')
        return
    first_dir = True
    res = []
    for i, source_dir in enumerate(source_dirs):
        print(f'Calculating likelihood {i + 1} / {n_dirs} at {time.strftime("%c")}', flush=True)

        # Extract cosmological parameters
        params = [None]*n_params
        values_path = os.path.join(source_dir, 'cosmological_parameters/values.txt')
        with open(values_path) as f:
            for line in f:
                for param_idx, param in enumerate(varied_params):
                    param_str = f'{param} = '
                    if param_str in line:
                        params[param_idx] = float(line[len(param_str):])
        err_str = f'Not all parameters in varied_params found in {values_path}'
        assert np.all([param is not None for param in params]), err_str

        # Check the ells for consistency
        if first_dir:
            ells_pos = np.loadtxt(os.path.join(source_dir, 'galaxy_cl/ell.txt'))
            ells_she = np.loadtxt(os.path.join(source_dir, 'shear_cl/ell.txt'))
            ells_pos_she = np.loadtxt(os.path.join(source_dir, 'galaxy_shear_cl/ell.txt'))
            assert np.allclose(ells_pos, ells_she)
            assert np.allclose(ells_pos, ells_pos_she)
            first_dir = False
        else:
            ells_pos_test = np.loadtxt(os.path.join(source_dir, 'galaxy_cl/ell.txt'))
            ells_she_test = np.loadtxt(os.path.join(source_dir, 'shear_cl/ell.txt'))
            ells_pos_she_test = np.loadtxt(os.path.join(source_dir, 'galaxy_shear_cl/ell.txt'))
            assert np.allclose(ells_pos, ells_pos_test)
            assert np.allclose(ells_she, ells_she_test)
            assert np.allclose(ells_pos_she, ells_pos_she_test)

        # Load theory Cls
        pos_pos_dir = os.path.join(source_dir, 'galaxy_cl/')
        she_she_dir = os.path.join(source_dir, 'shear_cl/')
        pos_she_dir = os.path.join(source_dir, 'galaxy_shear_cl/')
        theory_cls = like_w.load_cls(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir)

        # Evaluate likelihood
        log_like_wish = like_w.execute(ells_pos, theory_cls, config_w)

        # Store cosmological params & likelihood
        res.append([*params, log_like_wish])

    # Save results to file
    res_grid = np.asarray(res)
    param_names = ' '.join(varied_params)
    header = f'Output from {__file__}.run_like_cl_wishart for input grid {grid_dir} at {time.strftime("%c")} \n'
    header += f'{param_names} log_like_wishart'
    np.savetxt(save_path, res_grid, header=header)
    print('Saved ' + save_path)

    print(f'Done at {time.strftime("%c")}', flush=True)


def run_like_cl_gauss(grid_dir, varied_params, save_path, n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir,
                      pos_nl_path, she_nl_path, noise_ell_path, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax,
                      leff_path=None):
    """
    Evaluate Gaussian likelhood on a CosmoSIS output grid and save the result as a text file.

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        varied_params (list): List of varied parameter names as they appear in the cosmological_parameters/values.txt
                              file.
        save_path (str): Path to save output to.
        n_zbin (int): Number of redshift bins. It will be assumed that there is one position field and one shear field
                      per redshift bin.
        obs_pos_pos_dir (str): Path to the directory containing the observed position-position power spectra.
        obs_she_she_dir (str): Path to the directory containing the observed shear-shear power spectra.
        obs_pos_she_dir (str): Path to the directory containing the observed position-shear power spectra.
        pos_nl_path (str): Path to the position noise power spectrum.
        she_nl_path (str): Path to the shear noise power spectrum.
        noise_ell_path (str): Path to the file containing the ells for the noise power spectra.
        fid_pos_pos_dir (str): Path to the directory containing the fiducial theory position-position power spectra,
                               used for the covariance.
        fid_she_she_dir (str): Path to the directory containing the fiducial theory shear-shear power spectra.
        fid_pos_she_dir (str): Path to the directory containing the fiducial theory position-shear power spectra.
        lmax (int): Maximum l to use in the likelihood.
        leff_path (str, optional): Path to ell-ell_effective mapping, to replace each l with its corresponding l_eff
                                   when calculating the covariance.
    """

    print(f'Starting at {time.strftime("%c")}')

    # Setup the likelihood module
    print(f'Setting up likelihood module at {time.strftime("%c")}')
    config = like_g.setup(n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir, pos_nl_path, she_nl_path,
                          noise_ell_path, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax, leff_path=leff_path)
    print(f'Setup complete at {time.strftime("%c")}')

    # Loop over every input directory
    source_dirs = glob.glob(os.path.join(grid_dir, '_[0-9]*/'))
    n_dirs = len(source_dirs)
    if n_dirs == 0:
        warnings.warn(f'No matching directories. Terminating at {time.strftime("%c")}')
        return
    n_params = len(varied_params)
    if n_params == 0:
        warnings.warn(f'No parameters specified. Terminating at {time.strftime("%c")}')
        return
    first_dir = True
    res = []
    for i, source_dir in enumerate(source_dirs):
        print(f'Calculating likelihood {i + 1} / {n_dirs} at {time.strftime("%c")}')

        # Extract cosmological parameters
        params = [None]*n_params
        values_path = os.path.join(source_dir, 'cosmological_parameters/values.txt')
        with open(values_path) as f:
            for line in f:
                for param_idx, param in enumerate(varied_params):
                    param_str = f'{param} = '
                    if param_str in line:
                        params[param_idx] = float(line[len(param_str):])
        err_str = f'Not all parameters in varied_params found in {values_path}'
        assert np.all([param is not None for param in params]), err_str

        # Check the ells for consistency
        if first_dir:
            ells_pos = np.loadtxt(os.path.join(source_dir, 'galaxy_cl/ell.txt'))
            ells_she = np.loadtxt(os.path.join(source_dir, 'shear_cl/ell.txt'))
            ells_pos_she = np.loadtxt(os.path.join(source_dir, 'galaxy_shear_cl/ell.txt'))
            assert np.allclose(ells_pos, ells_she)
            assert np.allclose(ells_pos, ells_pos_she)
            first_dir = False
        else:
            ells_pos_test = np.loadtxt(os.path.join(source_dir, 'galaxy_cl/ell.txt'))
            ells_she_test = np.loadtxt(os.path.join(source_dir, 'shear_cl/ell.txt'))
            ells_pos_she_test = np.loadtxt(os.path.join(source_dir, 'galaxy_shear_cl/ell.txt'))
            assert np.allclose(ells_pos, ells_pos_test)
            assert np.allclose(ells_she, ells_she_test)
            assert np.allclose(ells_pos_she, ells_pos_she_test)

        # Load theory Cls
        pos_pos_dir = os.path.join(source_dir, 'galaxy_cl/')
        she_she_dir = os.path.join(source_dir, 'shear_cl/')
        pos_she_dir = os.path.join(source_dir, 'galaxy_shear_cl/')
        theory_cls = like_w.load_cls(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir)

        # Evaluate likelihood
        log_like_gauss = like_g.execute(ells_pos, theory_cls, config)

        # Store cosmological params & likelihood
        res.append([*params, log_like_gauss])

    # Save results to file
    res_grid = np.asarray(res)
    param_names = ' '.join(varied_params)
    header = f'Output from {__file__}.run_like_cl_gauss for input grid {grid_dir} at {time.strftime("%c")} \n'
    header += f'{param_names} log_like_gauss'
    np.savetxt(save_path, res_grid, header=header)
    print('Saved ' + save_path)

    print(f'Done at {time.strftime("%c")}')


def run_likes_cl_wishart_gauss(grid_dir, varied_params, save_path, n_zbin, obs_pos_pos_dir, obs_she_she_dir,
                               obs_pos_she_dir, pos_nl_path, she_nl_path, noise_ell_path, fid_pos_pos_dir,
                               fid_she_she_dir, fid_pos_she_dir, lmax, leff_path=None):
    """
    Evaluate both the Wishart and Gaussian likelhoods on a CosmoSIS output grid and save the result as a text file.

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        varied_params (list): List of varied parameter names as they appear in the cosmological_parameters/values.txt
                              file.
        save_path (str): Path to save output to.
        n_zbin (int): Number of redshift bins. It will be assumed that there is one position field and one shear field
                      per redshift bin.
        obs_pos_pos_dir (str): Path to the directory containing the observed position-position power spectra.
        obs_she_she_dir (str): Path to the directory containing the observed shear-shear power spectra.
        obs_pos_she_dir (str): Path to the directory containing the observed position-shear power spectra.
        pos_nl_path (str): Path to the position noise power spectrum.
        she_nl_path (str): Path to the shear noise power spectrum.
        noise_ell_path (str): Path to the file containing the ells for the noise power spectra.
        fid_pos_pos_dir (str): Path to the directory containing the fiducial theory position-position power spectra,
                               used for the covariance.
        fid_she_she_dir (str): Path to the directory containing the fiducial theory shear-shear power spectra.
        fid_pos_she_dir (str): Path to the directory containing the fiducial theory position-shear power spectra.
        lmax (int): Maximum l to use in the likelihood.
        leff_path (str, optional): Path to ell-ell_effective mapping, to replace each l with its corresponding l_eff
                                   when calculating the covariance.
    """

    print(f'Starting at {time.strftime("%c")}')

    # Setup the likelihood modules
    print(f'Setting up likelihood module 1/2 at {time.strftime("%c")}')
    config_w = like_w.setup(n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir, pos_nl_path, she_nl_path,
                            noise_ell_path, lmax, leff_path=leff_path)
    print(f'Setting up likelihood module 2/2 at {time.strftime("%c")}')
    config_g = like_g.setup(n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir, pos_nl_path, she_nl_path,
                            noise_ell_path, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax,
                            leff_path=leff_path)
    print(f'Setup complete at {time.strftime("%c")}')

    # Loop over every input directory
    source_dirs = glob.glob(os.path.join(grid_dir, '_[0-9]*/'))
    n_dirs = len(source_dirs)
    if n_dirs == 0:
        warnings.warn(f'No matching directories. Terminating at {time.strftime("%c")}')
        return
    n_params = len(varied_params)
    if n_params == 0:
        warnings.warn(f'No parameters specified. Terminating at {time.strftime("%c")}')
        return
    first_dir = True
    res = []
    for i, source_dir in enumerate(source_dirs):
        print(f'Calculating likelihood {i + 1} / {n_dirs} at {time.strftime("%c")}', flush=True)

        # Extract cosmological parameters
        params = [None]*n_params
        values_path = os.path.join(source_dir, 'cosmological_parameters/values.txt')
        with open(values_path) as f:
            for line in f:
                for param_idx, param in enumerate(varied_params):
                    param_str = f'{param} = '
                    if param_str in line:
                        params[param_idx] = float(line[len(param_str):])
        err_str = f'Not all parameters in varied_params found in {values_path}'
        assert np.all([param is not None for param in params]), err_str

        # Check the ells for consistency
        if first_dir:
            ells_pos = np.loadtxt(os.path.join(source_dir, 'galaxy_cl/ell.txt'))
            ells_she = np.loadtxt(os.path.join(source_dir, 'shear_cl/ell.txt'))
            ells_pos_she = np.loadtxt(os.path.join(source_dir, 'galaxy_shear_cl/ell.txt'))
            assert np.allclose(ells_pos, ells_she)
            assert np.allclose(ells_pos, ells_pos_she)
            first_dir = False
        else:
            ells_pos_test = np.loadtxt(os.path.join(source_dir, 'galaxy_cl/ell.txt'))
            ells_she_test = np.loadtxt(os.path.join(source_dir, 'shear_cl/ell.txt'))
            ells_pos_she_test = np.loadtxt(os.path.join(source_dir, 'galaxy_shear_cl/ell.txt'))
            assert np.allclose(ells_pos, ells_pos_test)
            assert np.allclose(ells_she, ells_she_test)
            assert np.allclose(ells_pos_she, ells_pos_she_test)

        # Load theory Cls
        pos_pos_dir = os.path.join(source_dir, 'galaxy_cl/')
        she_she_dir = os.path.join(source_dir, 'shear_cl/')
        pos_she_dir = os.path.join(source_dir, 'galaxy_shear_cl/')
        theory_cls = like_w.load_cls(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir)

        # Evaluate likelihoods
        log_like_wish = like_w.execute(ells_pos, theory_cls, config_w)
        log_like_gauss = like_g.execute(ells_pos, theory_cls, config_g)

        # Store cosmological params & likelihood
        res.append([*params, log_like_wish, log_like_gauss])

    print()

    # Save results to file
    res_grid = np.asarray(res)
    param_names = ' '.join(varied_params)
    header = f'Output from {__file__}.run_likes_cl_wishart_gauss for input grid {grid_dir} at {time.strftime("%c")} \n'
    header += f'{param_names} log_like_wishart log_like_gauss'
    np.savetxt(save_path, res_grid, header=header)
    print('Saved ' + save_path)

    print(f'Done at {time.strftime("%c")}', flush=True)


def max_like_1d(theory_filemask, obs_filemask, varied_param, save_dir, batch_size, n_zbin, fid_dir, pos_subdir,
                she_subdir, pos_she_subdir, ell_filename, pos_nl_path, she_nl_path, noise_ell_path, lmax, lmin):
    """
    Run Wishart and Gaussian likelihoods for a single parameter repeatedly for many observations,
    and save a text file with the maximum likelihood parameter values.

    Designed to process output from simulation.sim_cls_fullsky.

    Args:
        theory_filemask (str): Filemask for subdirectories of a 1D CosmoSIS output grid to iterate over.
        obs_filemask (str): Filemask for observations to iterate over.
        varied_param (str): Name of the parameter being varied, as it appears in the cosmological_parameters/values.txt
                            file.
        save_dir (str): Path to directory to save output to.
        batch_size (int): Number of observations per batch, where one text file is output per batch.
        n_zbin (int): Number of redshift bins.
        fid_dir (str): Path to directory containing fiducial power spectra, for Gaussian covariance.
        pos_subdir (str): Name of the sub-directory containing position-position power spectra.
        she_subdir (str): Name of the sub-directory containing shear-shear power spectra.
        pos_she_subdir (str): Name of the sub-directory containing position-shear power spectra.
        ell_filename (str): Filename containing ells within each sub-directory.
        pos_nl_path (str): Path to position noise power spectrum.
        she_nl_path (str): Path to shear noise power spectrum.
        noise_ell_path (str): Path to noise ells.
        lmax (int): Maximum ell for likelihood.
        lmin (int): Minimum ell for likelihood.
    """

    # Calculate some useful quantities
    n_field = 2 * n_zbin
    n_spec = n_field * (n_field + 1) // 2
    n_ell = lmax - lmin + 1

    # Calculate ell range
    ell = np.arange(lmin, lmax + 1)

    # Load fiducial ells and Cls, and trim to correct ell range
    fid_pos_pos_dir = os.path.join(fid_dir, pos_subdir)
    fid_she_she_dir = os.path.join(fid_dir, she_subdir)
    fid_pos_she_dir = os.path.join(fid_dir, pos_she_subdir)

    # Check ells for consistency
    fid_pos_ell = np.loadtxt(os.path.join(fid_pos_pos_dir, ell_filename))
    fid_she_ell = np.loadtxt(os.path.join(fid_she_she_dir, ell_filename))
    fid_pos_she_ell = np.loadtxt(os.path.join(fid_pos_she_dir, ell_filename))
    assert np.allclose(fid_pos_ell, fid_she_ell)
    assert np.allclose(fid_pos_ell, fid_pos_she_ell)
    assert min(fid_pos_ell) <= lmin
    assert max(fid_pos_ell) >= lmax

    # Load fiducial Cls and trim to correct ell range
    fid_lmin_in = int(min(fid_pos_ell))
    fid_cl = like_w.load_cls(n_zbin, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax, fid_lmin_in)[:, lmin:]
    assert fid_cl.shape == (n_spec, n_ell)

    # Calculate scaling factor used for numerical precision and apply to fiducial Cls
    scaling = 10 ** (2 + np.ceil(-np.log10(np.amax(fid_cl))))
    fid_cl *= scaling

    # Load noise ells and cls, and trim to correct ell range
    noise_ell = np.loadtxt(noise_ell_path)
    pos_nl = np.loadtxt(pos_nl_path)
    she_nl = np.loadtxt(she_nl_path)
    assert len(noise_ell) == len(pos_nl)
    assert len(noise_ell) == len(she_nl)
    assert min(noise_ell) <= lmin
    assert max(noise_ell) >= lmax
    nl_idxs = (noise_ell >= lmin) & (noise_ell <= lmax)
    pos_nl = pos_nl[nl_idxs]
    she_nl = she_nl[nl_idxs]

    # Convert noise cls to matrices, nonzero only for auto-spectra
    pos_nl *= scaling
    she_nl *= scaling
    nl_nonzero = np.array([pos_nl, she_nl]*n_zbin)
    nl_zero = np.zeros((n_spec - n_field, n_ell))
    nl = np.concatenate((nl_nonzero, nl_zero))
    nl_mats = like_w.cl_matrix(nl, n_field)
    nl = nl.t

    # Calculate inverse covariance matrix from fiducial cls + noise
    inv_cov = like_g.cl_invcov(ell, fid_cl.t, nl, n_field)
    assert inv_cov.shape == (n_ell, n_spec, n_spec)

    # Load theory cls and parameter values
    theory_dirs = glob.glob(theory_filemask)
    n_theory = len(theory_dirs)
    param_vals = np.full(n_theory, np.nan)
    param_str = f'{varied_param} = '
    theory_cls = np.full((n_theory, n_ell, n_spec), np.nan)
    theory_mats = np.full((n_theory, n_ell, n_field, n_field), np.nan)
    for theory_idx, theory_dir in enumerate(theory_dirs):

        pos_pos_dir = os.path.join(theory_dir, pos_subdir)
        she_she_dir = os.path.join(theory_dir, she_subdir)
        pos_she_dir = os.path.join(theory_dir, pos_she_subdir)

        # Extract parameter value
        with open(os.path.join(theory_dir, 'cosmological_parameters/values.txt')) as f:
            for line in f:
                if param_str in line:
                    param_vals[theory_idx] = float(line[len(param_str):])
                    break

        # Check ells for consistency
        pos_ell = np.loadtxt(os.path.join(pos_pos_dir, ell_filename))
        she_ell = np.loadtxt(os.path.join(she_she_dir, ell_filename))
        pos_she_ell = np.loadtxt(os.path.join(pos_she_dir, ell_filename))
        assert np.allclose(pos_ell, she_ell)
        assert np.allclose(pos_ell, pos_she_ell)
        assert min(pos_ell) <= lmin
        assert max(pos_ell) >= lmax

        # Load theory cls, trim to correct ell range and convert to matrices
        lmin_in = int(min(pos_ell))
        theory_cl = like_w.load_cls(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, lmax, lmin_in)[:, lmin:]
        assert theory_cl.shape == (n_spec, n_ell)
        theory_cl *= scaling
        theory_cls[theory_idx] = theory_cl.t
        theory_mats[theory_idx] = like_w.cl_matrix(theory_cl, n_field)

    assert np.all(np.isfinite(param_vals))
    assert np.all(np.isfinite(theory_cls))
    assert np.all(np.isfinite(theory_mats))

    # Loop over obs files matching filemask
    obs_file_paths = glob.glob(obs_filemask)
    n_obs_file = len(obs_file_paths)
    for obs_file_idx, obs_file_path in enumerate(obs_file_paths):
        print(f'opening obs file {obs_file_idx} / {n_obs_file} at {time.strftime("%c")}')

        # Load file and apply scaling
        with np.load(obs_file_path) as obs_file:
            obs_cl = obs_file['obs_cl']
        n_real = obs_cl.shape[0]
        assert obs_cl.shape[1:] == (n_spec, n_ell)
        obs_cl *= scaling

        n_batch = n_real // batch_size
        assert n_real % n_batch == 0

        # Loop over observations within batches
        for batch_idx in range(n_batch):
            print(f'obs file {obs_file_idx}: starting batch {batch_idx} / {n_batch} at {time.strftime("%c")}')
            maxlike_paramval_w = np.full(batch_size, np.nan)
            maxlike_paramval_g = np.full(batch_size, np.nan)
            for obs_idx_within_batch in range(batch_size):
                obs_idx = batch_idx * batch_size + obs_idx_within_batch
                print(f'obs file {obs_file_idx}; batch {batch_idx}: starting obs {obs_idx_within_batch} / {batch_size}'
                      f' at {time.strftime("%c")}')

                # Calculate obs Cl matrices
                obs_mats = like_w.cl_matrix(obs_cl[obs_idx], n_field)

                # Loop over theory directories and calculate the two log-likelihoods
                loglike_w = np.full(n_theory, np.nan)
                loglike_g = np.full(n_theory, np.nan)
                for i, (theory_cl, theory_mat) in enumerate(zip(theory_cls, theory_mats)):
                    loglike_w[i] = like_w.joint_log_likelihood(ell, theory_mat, nl_mats, obs_mats, lmax)
                    loglike_g[i] = like_g.joint_log_likelihood(ell, theory_cl, nl, obs_cl[obs_idx].t, inv_cov, lmax)
                assert np.all(np.isfinite(loglike_w))
                assert np.all(np.isfinite(loglike_g))

                # Find the param value corresponding to max of each likelihood
                maxlike_paramval_w[obs_idx_within_batch] = param_vals[np.argmax(loglike_w)]
                maxlike_paramval_g[obs_idx_within_batch] = param_vals[np.argmax(loglike_g)]

            # Save batch results to file
            assert np.all(np.isfinite(maxlike_paramval_w))
            assert np.all(np.isfinite(maxlike_paramval_g))
            save_path = os.path.join(save_dir, f'{obs_file_idx}_{batch_idx}.txt')
            to_save = np.column_stack((maxlike_paramval_w, maxlike_paramval_g))
            header = 'wishart_posterior_max gaussian_posterior_max'
            np.savetxt(save_path, to_save, header=header)
            print(f'Saved {save_path} at {time.strftime("%c")}')

        print(f'Closing obs file {obs_file_idx} at {time.strftime("%c")}')

    print(f'Done at {time.strftime("%c")}')


def post_mean_std_1d(theory_filemask, obs_filemask, varied_param, save_dir, batch_size, n_zbin, fid_dir, pos_subdir,
                     she_subdir, pos_she_subdir, ell_filename, pos_nl_path, she_nl_path, noise_ell_path, lmax, lmin):
    """
    Run Wishart and Gaussian likelihoods for a single parameter repeatedly for many observations,
    and save a text file with the posterior mean and standard deviation for each observation.

    Designed to process output from simulation.sim_cls_fullsky.

    Args:
        theory_filemask (str): Filemask for subdirectories of a 1D CosmoSIS output grid to iterate over.
        obs_filemask (str): Filemask for observations to iterate over.
        varied_param (str): Name of the parameter being varied, as it appears in the cosmological_parameters/values.txt
                            file.
        save_dir (str): Path to directory to save output to.
        batch_size (int): Number of observations per batch, where one text file is output per batch.
        n_zbin (int): Number of redshift bins.
        fid_dir (str): Path to directory containing fiducial power spectra, for Gaussian covariance.
        pos_subdir (str): Name of the sub-directory containing position-position power spectra.
        she_subdir (str): Name of the sub-directory containing shear-shear power spectra.
        pos_she_subdir (str): Name of the sub-directory containing position-shear power spectra.
        ell_filename (str): Filename containing ells within each sub-directory.
        pos_nl_path (str): Path to position noise power spectrum.
        she_nl_path (str): Path to shear noise power spectrum.
        noise_ell_path (str): Path to noise ells.
        lmax (int): Maximum ell for likelihood.
        lmin (int): Minimum ell for likelihood.
    """

    # Calculate some useful quantities
    n_field = 2 * n_zbin
    n_spec = n_field * (n_field + 1) // 2
    n_ell = lmax - lmin + 1

    # Calculate ell range
    ell = np.arange(lmin, lmax + 1)

    # Load fiducial ells and Cls, and trim to correct ell range
    fid_pos_pos_dir = os.path.join(fid_dir, pos_subdir)
    fid_she_she_dir = os.path.join(fid_dir, she_subdir)
    fid_pos_she_dir = os.path.join(fid_dir, pos_she_subdir)

    # Check ells for consistency
    fid_pos_ell = np.loadtxt(os.path.join(fid_pos_pos_dir, ell_filename))
    fid_she_ell = np.loadtxt(os.path.join(fid_she_she_dir, ell_filename))
    fid_pos_she_ell = np.loadtxt(os.path.join(fid_pos_she_dir, ell_filename))
    assert np.allclose(fid_pos_ell, fid_she_ell)
    assert np.allclose(fid_pos_ell, fid_pos_she_ell)
    assert min(fid_pos_ell) <= lmin
    assert max(fid_pos_ell) >= lmax

    # Load fiducial Cls and trim to correct ell range
    fid_lmin_in = int(min(fid_pos_ell))
    fid_cl = like_w.load_cls(n_zbin, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax, fid_lmin_in)[:, lmin:]
    assert fid_cl.shape == (n_spec, n_ell)

    # Calculate scaling factor used for numerical precision and apply to fiducial Cls
    scaling = 10 ** (2 + np.ceil(-np.log10(np.amax(fid_cl))))
    fid_cl *= scaling

    # Load noise ells and cls, and trim to correct ell range
    noise_ell = np.loadtxt(noise_ell_path)
    pos_nl = np.loadtxt(pos_nl_path)
    she_nl = np.loadtxt(she_nl_path)
    assert len(noise_ell) == len(pos_nl)
    assert len(noise_ell) == len(she_nl)
    assert min(noise_ell) <= lmin
    assert max(noise_ell) >= lmax
    nl_idxs = (noise_ell >= lmin) & (noise_ell <= lmax)
    pos_nl = pos_nl[nl_idxs]
    she_nl = she_nl[nl_idxs]

    # Convert noise cls to matrices, nonzero only for auto-spectra
    pos_nl *= scaling
    she_nl *= scaling
    nl_nonzero = np.array([pos_nl, she_nl]*n_zbin)
    nl_zero = np.zeros((n_spec - n_field, n_ell))
    nl = np.concatenate((nl_nonzero, nl_zero))
    nl_mats = like_w.cl_matrix(nl, n_field)
    nl = nl.t

    # Calculate inverse covariance matrix from fiducial cls + noise
    inv_cov = like_g.cl_invcov(ell, fid_cl.t, nl, n_field)
    assert inv_cov.shape == (n_ell, n_spec, n_spec)

    # Load theory cls and parameter values
    theory_dirs = glob.glob(theory_filemask)
    n_theory = len(theory_dirs)
    param_vals = np.full(n_theory, np.nan)
    param_str = f'{varied_param} = '
    theory_cls = np.full((n_theory, n_ell, n_spec), np.nan)
    theory_mats = np.full((n_theory, n_ell, n_field, n_field), np.nan)
    for theory_idx, theory_dir in enumerate(theory_dirs):

        pos_pos_dir = os.path.join(theory_dir, pos_subdir)
        she_she_dir = os.path.join(theory_dir, she_subdir)
        pos_she_dir = os.path.join(theory_dir, pos_she_subdir)

        # Extract parameter value
        with open(os.path.join(theory_dir, 'cosmological_parameters/values.txt')) as f:
            for line in f:
                if param_str in line:
                    param_vals[theory_idx] = float(line[len(param_str):])
                    break

        # Check ells for consistency
        pos_ell = np.loadtxt(os.path.join(pos_pos_dir, ell_filename))
        she_ell = np.loadtxt(os.path.join(she_she_dir, ell_filename))
        pos_she_ell = np.loadtxt(os.path.join(pos_she_dir, ell_filename))
        assert np.allclose(pos_ell, she_ell)
        assert np.allclose(pos_ell, pos_she_ell)
        assert min(pos_ell) <= lmin
        assert max(pos_ell) >= lmax

        # Load theory cls, trim to correct ell range and convert to matrices
        lmin_in = int(min(pos_ell))
        theory_cl = like_w.load_cls(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, lmax, lmin_in)[:, lmin:]
        assert theory_cl.shape == (n_spec, n_ell)
        theory_cl *= scaling
        theory_cls[theory_idx] = theory_cl.t
        theory_mats[theory_idx] = like_w.cl_matrix(theory_cl, n_field)

    assert np.all(np.isfinite(param_vals))
    assert np.all(np.isfinite(theory_cls))
    assert np.all(np.isfinite(theory_mats))

    # Calculate parameter step size, used to normalise posterior
    param_steps = np.diff(np.unique(param_vals))
    param_step = np.mean(param_steps)
    assert np.allclose(param_steps, param_step)

    # Loop over obs files matching filemask
    obs_file_paths = glob.glob(obs_filemask)
    n_obs_file = len(obs_file_paths)
    for obs_file_idx, obs_file_path in enumerate(obs_file_paths):
        print(f'opening obs file {obs_file_idx} / {n_obs_file} at {time.strftime("%c")}')

        # Load file and apply scaling
        with np.load(obs_file_path) as obs_file:
            obs_cl = obs_file['obs_cl']
        n_real = obs_cl.shape[0]
        assert obs_cl.shape[1:] == (n_spec, n_ell)
        obs_cl *= scaling

        n_batch = n_real // batch_size
        assert n_real % n_batch == 0

        # Loop over observations within batches
        for batch_idx in range(n_batch):
            print(f'obs file {obs_file_idx}: starting batch {batch_idx} / {n_batch} at {time.strftime("%c")}')
            means_w = np.full(batch_size, np.nan)
            means_g = np.full(batch_size, np.nan)
            stds_w = np.full(batch_size, np.nan)
            stds_g = np.full(batch_size, np.nan)
            for obs_idx_within_batch in range(batch_size):
                obs_idx = batch_idx * batch_size + obs_idx_within_batch
                print(f'obs file {obs_file_idx}; batch {batch_idx}: starting obs {obs_idx_within_batch} / {batch_size}'
                      f' at {time.strftime("%c")}')

                # Calculate obs Cl matrices
                obs_mats = like_w.cl_matrix(obs_cl[obs_idx], n_field)

                # Loop over theory directories and calculate the two log-likelihoods
                loglike_w = np.full(n_theory, np.nan)
                loglike_g = np.full(n_theory, np.nan)
                for i, (theory_cl, theory_mat) in enumerate(zip(theory_cls, theory_mats)):
                    loglike_w[i] = like_w.joint_log_likelihood(ell, theory_mat, nl_mats, obs_mats, lmax)
                    loglike_g[i] = like_g.joint_log_likelihood(ell, theory_cl, nl, obs_cl[obs_idx].t, inv_cov, lmax)
                assert np.all(np.isfinite(loglike_w))
                assert np.all(np.isfinite(loglike_g))

                # Convert log-likelihood to normalised posterior
                post_w = posteriors.log_like_to_post(loglike_w, param_step)
                post_g = posteriors.log_like_to_post(loglike_g, param_step)

                # Integrate to find the mean and variance
                mean_w = np.sum(param_vals * post_w) * param_step
                mean_g = np.sum(param_vals * post_g) * param_step
                var_w = np.sum(param_vals ** 2 * post_w) * param_step - mean_w ** 2
                var_g = np.sum(param_vals ** 2 * post_g) * param_step - mean_g ** 2

                # Store mean and standard deviation
                means_w[obs_idx_within_batch] = mean_w
                means_g[obs_idx_within_batch] = mean_g
                stds_w[obs_idx_within_batch] = np.sqrt(var_w)
                stds_g[obs_idx_within_batch] = np.sqrt(var_g)

            # Save batch results to file
            assert np.all(np.isfinite(means_w))
            assert np.all(np.isfinite(means_g))
            assert np.all(np.isfinite(stds_w))
            assert np.all(np.isfinite(stds_g))
            save_path = os.path.join(save_dir, f'{obs_file_idx}_{batch_idx}.txt')
            to_save = np.column_stack((means_w, means_g, stds_w, stds_g))
            header = 'mean_wishart mean_gauss std_wishart std_gauss'
            np.savetxt(save_path, to_save, header=header)
            print(f'Saved {save_path} at {time.strftime("%c")}')

        print(f'Closing obs file {obs_file_idx} at {time.strftime("%c")}')

    print(f'Done at {time.strftime("%c")}')
