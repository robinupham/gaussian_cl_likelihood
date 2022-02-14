"""
Contains functions relating to generating and processing simulations.
"""

import enum
import glob
import os.path
import pathlib
import time

import healpy as hp
import numpy as np
import scipy.linalg
import scipy.stats

# Workaround to get docs to compile
try:
    import pymaster as nmt
except ImportError:
    nmt = None

import gaussian_cl_likelihood.python.like_cl_wishart as like_w


class FieldType(enum.Enum):
    """
    The three types of fields: position, shear E and shear B.
    """
    POS = enum.auto()
    SHE_E = enum.auto()
    SHE_B = enum.auto()


def idx_to_field_type(idx):
    """
    Return whether a row or column index corresponds to position, shear E-mode or shear B-mode, based on the remainder
    when dividing by 3.

    Args:
        idx (int): Field index.

    Returns:
        FieldType: FieldType corresponding to field index.
    """
    remainder = idx % 3
    return FieldType.POS if remainder == 0 else (FieldType.SHE_E if remainder == 1 else FieldType.SHE_B)


def idx_to_field_type_nob(idx):
    """
    Alternative to idx_to_field_type for when shear B modes are not included.
    Return whether a row or column index corresponds to position or shear E-mode, based on the remainder when dividing
    by 2.

    Args:
        idx (int): Field index.

    Returns:
        FieldType: FieldType corresponding to field index.
    """

    remainder = idx % 2
    return FieldType.POS if remainder == 0 else FieldType.SHE_E


def idx_to_zbin(idx):
    """
    Return the z bin corresponding to a particular row or column index, assuming 3 fields per z bin.

    Args:
        idx (int): Field index.

    Returns:
        int: z-bin corresponding to field index, where 1 is the first redshift bin.
    """
    return 1 + idx // 3


def idx_to_zbin_nob(idx):
    """
    Alternative to idx_to_zbin for when shear B modes are not included.
    Return the z bin corresponding to a particular row or column index, assuming 2 fields per z bin.

    Args:
        idx (int): Field index.

    Returns:
        int: z-bin corresponding to field index, where 1 is the first redshift bin.
    """
    return 1 + idx // 2


def load_cls_zerob(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, lmax, lmin_in=0):
    """
    Given the number of redshift bins and relevant directories, load power spectra (position, shear, cross) in the
    correct order (diagonal / healpy new=True ordering).
    For lmin_in > 0 they will be padded with zeros, suitable for input to synfast.
    Assumes zero shear B-mode.

    Args:
        n_zbin (int): Number of redshift bins.
        pos_pos_dir (str): Path to directory containing position-position power spectra.
        she_she_dir (str): Path to directory containing shear-shear power spectra.
        pos_she_dir (str): Path to directory containing position-shear power spectra.
        lmax (int): Maximum l to load.
        lmin_in (int, optional): l corresponding to the first line in each input file, default 0.

    Returns:
        spectra (2D numpy array)
            Array of input Cls with shape (n_spectra, lmax + 1), where n_spectra = n * (n + 1) / 2
            and n = 3 * n_zbin. Padded with zeros below l = lmin_in.
        b_indices (list)
            Indices of spectra that involve shear B-modes, i.e. BB, EB, NB.
    """

    # Calculate number of fields assuming 1 position field and 1 spin-2 shear field per redshift bin
    n_field = 3 * n_zbin

    # Load power spectra in 'diagonal order'
    spectra = []
    b_indices = []
    i = -1
    for diag in range(n_field):
        for row in range(n_field - diag):
            col = row + diag
            i += 1

            # Determine the field types for this row and column
            row_type = idx_to_field_type(row)
            col_type = idx_to_field_type(col)

            # If either row or column is shear B-mode, the input spectrum is zero
            if row_type == FieldType.SHE_B or col_type == FieldType.SHE_B:
                spectra.append(np.zeros(lmax + 1))
                b_indices.append(i)
                continue

            # Determine the redshift bins for this row and column, and order them for cosmosis output:
            # for pos-pos and she-she the higher bin index goes first, for pos-she pos goes first
            bins = (idx_to_zbin(row), idx_to_zbin(col))
            if row_type == col_type: # pos-pos or she-she
                bin1 = max(bins)
                bin2 = min(bins)
            elif row_type == FieldType.POS: # pos-she
                bin1, bin2 = bins
            else: # she-pos, so invert
                bin2, bin1 = bins

            # Determine the input path
            if row_type == FieldType.POS and col_type == FieldType.POS:
                cl_dir = pos_pos_dir
            elif row_type == FieldType.SHE_E and col_type == FieldType.SHE_E:
                cl_dir = she_she_dir
            else:
                cl_dir = pos_she_dir
            cl_path = os.path.join(cl_dir, f'bin_{bin1}_{bin2}.txt')

            # Load with appropriate ell range
            spec = np.concatenate((np.zeros(lmin_in), np.loadtxt(cl_path, max_rows=(lmax - lmin_in + 1))))
            spectra.append(spec)

    assert len(spectra) - 1 == i, f'len(spectra) is {len(spectra)} but i is {i}'
    return np.asarray(spectra), b_indices


def get_binning_matrix(n_bandpowers, output_lmin, output_lmax, input_lmin=None, input_lmax=None):
    """
    Returns the binning matrix to convert Cls to log-spaced bandpowers, following Eqn. 20 of Hivon et al. 2002.

    Input ell range defaults to match output ell range - note this behaviour is not suitable if this matrix is to be
    used to multiply the raw output from healpy anafast/alm2cl, which returns all ells from l=0. In that case,
    explicitly set input_lmin=0.

    Args:
        n_bandpowers (int): Number of bandpowers required.
        output_lmin (int): Minimum l to include in the binning.
        output_lmax (int): Maximum l to include in the binning.
        input_lmin (int, optional): Minimum l in the input (defaults to output_lmin).
        input_lmax (int, optional): Maximum l in the input (defaults to output_lmax).

    Returns:
        2D numpy array: Binning matrix P_bl, shape (n_bandpowers, n_input_ell),
                        with n_input_ell = input_lmax - input_lmin + 1.
    """

    # Calculate bin boundaries (add small fraction to lmax to include it in the end bin)
    edges = np.logspace(np.log10(output_lmin), np.log10(output_lmax + 1e-5), n_bandpowers + 1)

    # Calculate input ell range and create broadcasted views for convenience
    if input_lmin is None:
        input_lmin = output_lmin
    if input_lmax is None:
        input_lmax = output_lmax
    ell = np.arange(input_lmin, input_lmax + 1)[None, ...]
    lower_edges = edges[:-1, None]
    upper_edges = edges[1:, None]

    # First calculate a boolean matrix of whether each ell is included in each bandpower,
    # then apply the l(l+1)/2π / n_l factor where n_l is the number of ells in the bin
    in_bin = (ell >= lower_edges) & (ell < upper_edges)
    n_ell = np.floor(upper_edges) - np.ceil(lower_edges) + 1
    pbl = in_bin * ell * (ell + 1) / (2 * np.pi * n_ell)

    return pbl


def save_maps(maps, save_dir, zbin_no):
    """
    Save a set of three maps (galaxy position, shear 1, shear 2) to a single file.

    Args:
        maps (sequence of 3 healpy maps): Maps of (galaxy position, shear 1, shear 2).
        save_dir (str): Path of directory to save maps to.
        zbin_no (int): Redshift bin number, used to label the output file.
    """

    save_path = os.path.join(save_dir, f'zbin{zbin_no}.fits')
    hp.fitsfunc.write_map(save_path, maps, column_names=['POSITION', 'SHEAR_1', 'SHEAR_2'], dtype=maps.dtype)
    print(f'Saved maps for z-bin {zbin_no} to {save_path}')


def save_cls_nob(spectra, n_zbin, save_dir, pos_pos_subdir='galaxy_cl', she_she_subdir='shear_cl',
                 pos_she_subdir='galaxy_shear_cl', lmin=None, verbose=True):
    """
    Takes a diagonal-ordered sequence of tomographic 3x2pt power spectra and saves them each to an individual file
    matching the structure of CosmoSIS output. Will also save ells if lmin is supplied.
    Only accounts for shear E-mode.

    Args:
        spectra (2D numpy array): Power spectra to save, with shape (n_spectra, n_ell).
        n_zbin (int): Number of redshift bins.
        save_dir (str): Path to base directory to save Cls to.
        pos_pos_subdir (str, optional): Name of subdirectory for position-position power spectra, default 'galaxy_cl'.
        she_she_subdir (str, optional): Name of subdirectory for shear-shear power spectra, default 'shear_cl'.
        pos_she_subdir (str, optional): Name of subdirectory for position-shear power spectra,
                                        default 'galaxy_shear_cl'.
        lmin (int, optional): Minimum ell to list in ells file, which is only produced if lmin is supplied.
        verbose (bool, optional): Whether to print names of files as they are saved (default True).
    """

    assert spectra.ndim == 2, 'Spectra must be in vector form, not matrix form'

    # Create output directories
    pos_pos_dir = os.path.join(save_dir, pos_pos_subdir)
    she_she_dir = os.path.join(save_dir, she_she_subdir)
    pos_she_dir = os.path.join(save_dir, pos_she_subdir)
    pathlib.Path(pos_pos_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(she_she_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(pos_she_dir).mkdir(parents=True, exist_ok=True)

    # Save ells if required
    if lmin is not None:
        n_ell = spectra.shape[1]
        lmax = lmin + n_ell - 1
        ell = np.arange(lmin, lmax + 1)
        for dir_ in (pos_pos_dir, she_she_dir, pos_she_dir):
            ell_path = os.path.join(dir_, 'ell.txt')
            np.savetxt(ell_path, ell, header='ell')
            if verbose:
                print('Saved ' + ell_path)

    # Calculate number of fields assuming 1 position field and 1 shear field per redshift bin
    n_field = 2 * n_zbin

    # Iterate through spectra and save
    spectrum_idx = 0
    for diag in range(n_field):
        for row in range(n_field - diag):
            col = row + diag

            # Determine the field types for this row and column
            row_type = idx_to_field_type_nob(row)
            col_type = idx_to_field_type_nob(col)

            # Determine the redshift bins for this row and column, and order them to match cosmosis output:
            # for pos-pos and she-she the higher bin index goes first, for pos-she pos goes first
            bins = (idx_to_zbin_nob(row), idx_to_zbin_nob(col))
            if row_type == col_type: # pos-pos or she-she
                bin1 = max(bins)
                bin2 = min(bins)
            elif row_type == FieldType.POS: # pos-she
                bin1, bin2 = bins
            else: # she-pos, so invert
                bin2, bin1 = bins

            # Determine the output path
            if row_type == FieldType.POS and col_type == FieldType.POS:
                cl_dir = pos_pos_dir
            elif row_type == FieldType.SHE_E and col_type == FieldType.SHE_E:
                cl_dir = she_she_dir
            else:
                cl_dir = pos_she_dir
            filename = f'bin_{bin1}_{bin2}'
            cl_path = os.path.join(cl_dir, filename + '.txt')

            # Save to file
            np.savetxt(cl_path, spectra[spectrum_idx], header=filename)
            if verbose:
                print(f'Saved spectrum {spectrum_idx} to {cl_path}')

            spectrum_idx += 1


def save_bandpowers(bandpowers, n_zbin, save_dir, pbl=None, pos_pos_subdir='galaxy_bp', she_she_subdir='shear_bp',
                    pos_she_subdir='galaxy_shear_bp', verbose=True):
    """
    Save bandpowers to disk with one text file per power spectrum. Also saves bandpower binning matrix pbl if provided.

    Args:
        bandpowers (2D numpy array): Binned power spectra to save, with shape (n_spectra, n_bandpower).
        n_zbin (int): Number of redshift bins.
        save_dir (str): Path to base directory to save bandpowers to.
        pbl (2D numpy array, optional): Bandpower binning matrix, which will be saved too if provided.
        pos_pos_subdir (str, optional): Name of subdirectory for position-position power spectra, default 'galaxy_bp'.
        she_she_subdir (str, optional): Name of subdirectory for shear-shear power spectra, default 'shear_bp'.
        pos_she_subdir (str, optional): Name of subdirectory for position-shear power spectra,
                                        default 'galaxy_shear_bp'.
        verbose (bool, optional): Whether to print names of files as they are saved (default True).
    """

    # Save Pbl matrix
    if pbl is not None:
        pbl_path = os.path.join(save_dir, 'pbl.txt')
        np.savetxt(pbl_path, pbl, header='Pbl binning matrix')
        if verbose:
            print('Saved Pbl binning matrix to ' + pbl_path)

    # Save bandpowers using save_cls_nob function
    save_cls_nob(bandpowers, n_zbin, save_dir, pos_pos_subdir, she_she_subdir, pos_she_subdir, None, verbose)


def single_obs_cls(n_zbin, pos_pos_in_dir, she_she_in_dir, pos_she_in_dir, lmax, lmin_in, pos_nl_path, she_nl_path,
                   nside, lmin_out, mask_path=None, verbose=True):
    """
    Generate a single cut-sky realisation of all power spectra.

    Args:
        n_zbin (int): Number of redshift bins.
        pos_pos_in_dir (str): Path to directory containing input position-position power spectra.
        she_she_in_dir (str): Path to directory containing input shear-shear power spectra.
        pos_she_in_dir (str): Path to directory containing input position-shear power spectra.
        lmax (int): Maximum l to simulate.
        lmin_in (int): l corresponding to first row in input files.
        pos_nl_path (str): Path to position noise power spectrum.
        she_nl_path (str): Path to shear noise power spectrum.
        nside (int): Healpix map resolution to use.
        lmin_out (int): Minimum l to return.
        mask_path (str, optional): Path to mask fits file, or None for full sky (default).
        verbose (bool, optional): Whether to output progress (default True).

    Returns:
        2D numpy array: Observed Cls in diagonal order, with shape (n_spectra, n_ell).
    """

    # Load all of the underlying power spectra in the correct order
    # and obtain the indices of the spectra involving B modes
    if verbose:
        print('Loading underlying Cls')
    spectra, _ = load_cls_zerob(n_zbin, pos_pos_in_dir, she_she_in_dir, pos_she_in_dir, lmax, lmin_in)

    # Load the noise power spectra
    if verbose:
        print('Loading noise Cls')
    pos_nl = np.concatenate((np.zeros(lmin_in), np.loadtxt(pos_nl_path, max_rows=(lmax - lmin_in + 1))))
    she_nl = np.concatenate((np.zeros(lmin_in), np.loadtxt(she_nl_path, max_rows=(lmax - lmin_in + 1))))

    # Do some consistency checks
    if verbose:
        print('Performing consistency checks')
    assert len(pos_nl) == len(she_nl)
    assert np.all([len(pos_nl) == len(spec) for spec in spectra])

    # Add noise to auto-spectra
    print('Adding noise')
    n_fields = 3 * n_zbin
    spectra[:n_fields:3] += pos_nl
    spectra[1:n_fields:3] += she_nl # E
    spectra[2:n_fields:3] += she_nl # B

    # Load the mask or use full sky
    if mask_path is not None:
        if verbose:
            print('Loading mask')
        mask = hp.pixelfunc.ud_grade(hp.fitsfunc.read_map(mask_path, dtype=None, verbose=False), nside)
    else:
        if verbose:
            print('Using full-sky mask')
        mask = np.ones(hp.pixelfunc.nside2npix(nside))

    # Calculate indices of fields to keep (not shear B-mode) and create array to hold observed bandpowers
    fields_to_keep = [field_idx for field_idx in range(n_fields) if idx_to_field_type(field_idx) is not FieldType.SHE_B]
    assert len(fields_to_keep) == 2 * n_zbin

    # Generate full-sky alms
    if verbose:
        print('Generating harmonic space realisation')
    alms_fullsky = hp.sphtfunc.synalm(spectra, lmax=lmax, new=True, verbose=False)

    # Mask in pixel space for each z bin independently
    alms_masked = np.empty_like(alms_fullsky)
    for z_idx, z_fs_alms in enumerate(np.split(alms_fullsky, n_zbin)):
        print(f'Applying mask for z bin {z_idx + 1} / {n_zbin}')
        maps = hp.sphtfunc.alm2map(z_fs_alms, nside, pol=True, verbose=False)
        maps *= mask
        alms_masked[(z_idx * 3):((z_idx + 1) * 3)] = hp.sphtfunc.map2alm(maps, lmax=lmax, pol=True)

    # Discard shear B-mode alms and calculate Cls
    print('Calculating observed Cls')
    obs_cls = hp.sphtfunc.alm2cl(alms_masked[fields_to_keep, :])[:, lmin_out:]

    return obs_cls


def sim_bps(n_zbin, pos_pos_in_dir, she_she_in_dir, pos_she_in_dir, lmax, lmin_in, pos_nl_path, she_nl_path,
            nside, n_bandpower, lmin_out, mask_path, n_loop, batch_size, save_dir):
    """
    Generate simultaneous full-sky and cut-sky repeated bandpower realisations, saved to disk in batches.

    Args:
        n_zbin (int): Number of redshift bins.
        pos_pos_in_dir (str): Path to directory containing input position-position power spectra.
        she_she_in_dir (str): Path to directory containing input shear-shear power spectra.
        pos_she_in_dir (str): Path to directory containing input position-shear power spectra.
        lmax (int): Maximum l to simulate.
        lmin_in (int): l corresponding to first row in input files.
        pos_nl_path (str): Path to position noise power spectrum.
        she_nl_path (str): Path to shear noise power spectrum.
        nside (int): Healpix map resolution to use.
        n_bandpower (int): Number of log-spaced bandpowers to use.
        lmin_out (int): Minimum l to include in bandpowers.
        mask_path (str): Path to mask fits file.
        n_loop (int): Total number of realisations to generate.
        batch_size (int): Number of realisations per batch, where one file is saved per batch.
        save_dir (str): Path to directory to save batches into (must already exist).
    """

    # Load all of the underlying power spectra in the correct order
    # and obtain the indices of the spectra involving B modes
    print('Loading underlying Cls')
    spectra, _ = load_cls_zerob(n_zbin, pos_pos_in_dir, she_she_in_dir, pos_she_in_dir, lmax, lmin_in)

    # Load the noise power spectra
    print('Loading noise Cls')
    pos_nl = np.concatenate((np.zeros(lmin_in), np.loadtxt(pos_nl_path, max_rows=(lmax - lmin_in + 1))))
    she_nl = np.concatenate((np.zeros(lmin_in), np.loadtxt(she_nl_path, max_rows=(lmax - lmin_in + 1))))

    # Do some consistency checks
    print('Performing consistency checks')
    assert len(pos_nl) == len(she_nl)
    assert np.all([len(pos_nl) == len(spec) for spec in spectra])

    # Add noise to auto-spectra
    print('Adding noise')
    n_fields = 3 * n_zbin
    spectra[:n_fields:3] += pos_nl
    spectra[1:n_fields:3] += she_nl # E
    spectra[2:n_fields:3] += she_nl # B

    # Load the mask
    print('Loading mask')
    mask = hp.pixelfunc.ud_grade(hp.fitsfunc.read_map(mask_path, dtype=None, verbose=False), nside)

    # Form the binning matrix
    print('Calculating binning matrix')
    pbl = get_binning_matrix(n_bandpower, lmin_out, lmax)

    # Calculate indices of fields to keep (not shear B-mode) and create arrays to hold observed bandpowers
    fields_to_keep = [field_idx for field_idx in range(n_fields) if idx_to_field_type(field_idx) is not FieldType.SHE_B]
    assert len(fields_to_keep) == 2 * n_zbin
    n_spectra_to_keep = len(fields_to_keep) * (len(fields_to_keep) + 1) // 2

    n_batch = n_loop // batch_size
    for batch in range(n_batch):
        start_time = time.time()
        print(f'Starting batch {batch} at: {time.strftime("%c")}')

        # Create arrays to hold observed bandpowers
        obs_bp_fs = np.full((batch_size, n_spectra_to_keep, n_bandpower), np.nan)
        obs_bp_ma = np.full((batch_size, n_spectra_to_keep, n_bandpower), np.nan)

        for i in range(batch_size):
            print(f'Batch {batch}: realisation {i + 1} / {batch_size}', flush=True) #, end='\r')

            # Generate full-sky alms
            alms_fullsky = hp.sphtfunc.synalm(spectra, lmax=lmax, new=True, verbose=False)

            # Mask in pixel space for each z bin independently
            alms_masked = np.empty_like(alms_fullsky)
            for z_idx, z_fs_alms in enumerate(np.split(alms_fullsky, n_zbin)):
                maps = hp.sphtfunc.alm2map(z_fs_alms, nside, pol=True, verbose=False)
                maps *= mask
                alms_masked[(z_idx * 3):((z_idx + 1) * 3)] = hp.sphtfunc.map2alm(maps, lmax=lmax, pol=True)

            # Discard shear B-mode alms and calculate Cls
            obs_cls_fs = hp.sphtfunc.alm2cl(alms_fullsky[fields_to_keep, :])[:, lmin_out:]
            obs_cls_ma = hp.sphtfunc.alm2cl(alms_masked[fields_to_keep, :])[:, lmin_out:]

            # Apply the binning matrix to each observed power spectrum to calculate bandpowers
            obs_bp_fs[i] = np.einsum('bl,sl->sb', pbl, obs_cls_fs)
            obs_bp_ma[i] = np.einsum('bl,sl->sb', pbl, obs_cls_ma)

        stop_time = time.time()
        print(f'Batch {batch}: Time taken for {n_batch} realisations: {(stop_time - start_time):.1f} s')

        assert np.all(np.isfinite(obs_bp_fs))
        assert np.all(np.isfinite(obs_bp_ma))
        save_path_fs = os.path.join(save_dir, f'fs_{batch}.npz')
        save_path_ma = os.path.join(save_dir, f'ma_{batch}.npz')
        np.savez_compressed(save_path_fs, obs_bp=obs_bp_fs)
        print('Saved ' + save_path_fs)
        np.savez_compressed(save_path_ma, obs_bp=obs_bp_ma)
        print('Saved ' + save_path_ma)
        print(f'Ending batch {batch} at: {time.strftime("%c")}')
        print()

    print(f'Done at: {time.strftime("%c")}')


def sim_cls_fullsky(n_zbin, pos_pos_in_dir, she_she_in_dir, pos_she_in_dir, lmax, lmin_in, pos_nl_path, she_nl_path,
                    lmin_out, n_loop, batch_size, save_dir):
    """
    Generate full-sky repeated power spectrum realisations, saved to disk in batches.

    Args:
        n_zbin (int): Number of redshift bins.
        pos_pos_in_dir (str): Path to directory containing input position-position power spectra.
        she_she_in_dir (str): Path to directory containing input shear-shear power spectra.
        pos_she_in_dir (str): Path to directory containing input position-shear power spectra.
        lmax (int): Maximum l to simulate.
        lmin_in (int): l corresponding to first row in input files.
        pos_nl_path (str): Path to position noise power spectrum.
        she_nl_path (str): Path to shear noise power spectrum.
        lmin_out (int): Minimum l to include in output.
        n_loop (int): Total number of realisations to generate.
        batch_size (int): Number of realisations per batch, where one file is saved per batch.
        save_dir (str): Path to directory to save batches into (must already exist).
    """

    # Load all of the underlying power spectra in the correct order
    # and obtain the indices of the spectra involving B modes
    print('Loading underlying Cls')
    spectra, _ = load_cls_zerob(n_zbin, pos_pos_in_dir, she_she_in_dir, pos_she_in_dir, lmax, lmin_in)

    # Load the noise power spectra
    print('Loading noise Cls')
    pos_nl = np.concatenate((np.zeros(lmin_in), np.loadtxt(pos_nl_path, max_rows=(lmax - lmin_in + 1))))
    she_nl = np.concatenate((np.zeros(lmin_in), np.loadtxt(she_nl_path, max_rows=(lmax - lmin_in + 1))))

    # Do some consistency checks
    print('Performing consistency checks')
    assert len(pos_nl) == len(she_nl)
    assert np.all([len(pos_nl) == len(spec) for spec in spectra])

    # Add noise to auto-spectra
    print('Adding noise')
    n_fields = 3 * n_zbin
    spectra[:n_fields:3] += pos_nl
    spectra[1:n_fields:3] += she_nl # E
    spectra[2:n_fields:3] += she_nl # B

    # Calculate indices of fields to keep (not shear B-mode)
    fields_to_keep = [field_idx for field_idx in range(n_fields) if idx_to_field_type(field_idx) is not FieldType.SHE_B]
    assert len(fields_to_keep) == 2 * n_zbin
    n_spectra_to_keep = len(fields_to_keep) * (len(fields_to_keep) + 1) // 2

    n_batch = n_loop // batch_size
    for batch in range(n_batch):
        start_time = time.time()
        print(f'Starting batch {batch} at: {time.strftime("%c")}')

        # Create arrays to hold observed Cls
        obs_cl = np.full((batch_size, n_spectra_to_keep, lmax - lmin_out + 1), np.nan)

        for i in range(batch_size):
            print(f'Batch {batch}: realisation {i + 1} / {batch_size}', flush=True)

            # Generate alms
            alms = hp.sphtfunc.synalm(spectra, lmax=lmax, new=True, verbose=False)

            # Discard shear B-mode alms and calculate Cls
            obs_cl[i] = hp.sphtfunc.alm2cl(alms[fields_to_keep, :])[:, lmin_out:]

        stop_time = time.time()
        print(f'Batch {batch}: Time taken for {batch_size} realisations: {(stop_time - start_time):.1f} s')

        assert np.all(np.isfinite(obs_cl))
        save_path = os.path.join(save_dir, f'{batch}.npz')
        np.savez_compressed(save_path, obs_cl=obs_cl)
        print('Saved ' + save_path)
        print(f'Ending batch {batch} at: {time.strftime("%c")}')
        print()

    print(f'Done at: {time.strftime("%c")}')


def _check_nmt():
    if nmt is None:
        raise ImportError('pymaster must be installed. Please see '
                          'https://namaster.readthedocs.io/en/latest/installation.html')


def flat_bins_linspaced(leff_min, leff_max, bin_size):
    """
    Returns a NaMaster NmtBinFlat object whose effective ells are linearly spaced from leff_min to leff_max (inclusive)
    with a step of bin_size.

    Args:
        leff_min (float): Minimum effective ell.
        leff_max (float): Maximum effective ell.
        bin_size (float): Difference between effective ells for adjacent bins.

    Returns:
        NmtBinFlat: NaMaster flat binning scheme.
    """
    _check_nmt()

    l0 = np.arange(leff_min - bin_size / 2, leff_max + bin_size / 2, bin_size)
    lf = l0 + bin_size
    bp = nmt.NmtBinFlat(l0, lf)
    assert np.allclose(bp.get_effective_ells(), np.arange(leff_min, leff_max + 1, bin_size))
    return bp


def sim_flat(n_zbin, cl_filemask, input_lmin, nx, ny, lx, ly, leff_min, leff_max, bin_size, n_real, save_path):
    """
    Generate repeated flat-sky realisations and save measured bandpowers to disk.

    Args:
        n_zbin (int): Number of redshift bins.
        cl_filemask (str): Path to input Cls, with the two redshift bins replaced with {i} and {j}.
        input_lmin (int): l corresponding to first line in input files.
        nx (int): Number of pixels along the x axis.
        ny (int): Number of pixels along the y axis.
        lx (float): Size along the x axis in radians.
        ly (float): Size along the y axis in radians.
        leff_min (float): Effective l for the lowest bandpower.
        leff_max (float): Effective l for the highest bandpower.
        bin_size (float): Difference between effective l for adjacent bandpowers.
        n_real (int): Number of realisations to generate.
        save_path (str): Path to save output to.
    """
    _check_nmt()

    # Load input power spectra in the order required by synfast_flat
    cls_ = []
    for i in range(1, n_zbin + 1):
        for j in range(i, n_zbin + 1):
            cl_path = cl_filemask.format(i=i, j=j)
            cls_.append(np.concatenate((np.zeros(input_lmin), np.loadtxt(cl_path))))

    # Other things that don't change per realisation
    spin = np.zeros(n_zbin, dtype=int)
    mask = np.ones((nx, ny))
    bins_ = flat_bins_linspaced(leff_min, leff_max, bin_size)
    leff = bins_.get_effective_ells()
    nbin = len(leff)
    n_spec = len(cls_)

    # Generate realisations
    bps = np.full((n_real, n_spec, nbin), np.nan)
    for i in range(n_real):
        print(f'{i + 1} / {n_real}', end='\r')

        maps = nmt.synfast_flat(nx, ny, lx, ly, cls_, spin)
        fields = [nmt.nmtfieldflat(lx, ly, mask, [map_]) for map_ in maps]

        # Calculate raw (not deconvolved) observed bandpowers
        spec_idx = 0
        for field1 in range(n_zbin):
            for field2 in range(field1, n_zbin):
                bps[i, spec_idx] = np.squeeze(nmt.compute_coupled_cell_flat(fields[field1], fields[field2], bins_))
                spec_idx += 1
    assert np.all(np.isfinite(bps))

    # Save to disk
    header = (f'Output from {__file__}.sim_flat; {n_real} realisations of raw (not deconvolved) bandpowers; '
              f'{time.strftime("%c")}')
    np.savez_compressed(save_path, leff=leff, bps=bps, header=header)
    print('Saved ' + save_path)


def is_pd(matrix):
    """
    Test whether a matrix is positive definite, using Scipy's Cholesky decomposition.

    Args:
        matrix (2D numpy array): Matrix to test.
    Returns:
        bool: True if positive definite, False otherwise.
    """
    try:
        _ = scipy.linalg.cholesky(matrix)
        return True
    except scipy.linalg.LinAlgError:
        return False


def combine_sim_bp_batches(input_mask):
    """
    Combine batches produced by sim_bps into a single file.

    Args:
        input_mask (str): Path to batches, with batch number replaced by {batch}.
    """

    # Load all batches
    batches = []
    batch = 0
    while os.path.exists(input_mask.format(batch=batch)):
        print(f'Loading batch {batch}', end='\r')
        with np.load(input_mask.format(batch=batch)) as data:
            batches.append(data['obs_bp'])
        batch += 1
    print(f'{len(batches)} batches loaded')

    # Combine into a single file
    combined = np.concatenate(batches)
    save_path = input_mask.format(batch='combined')
    header = f'Output from {__file__} for input {input_mask}, date {time.strftime("%c")}'
    np.savez_compressed(save_path, obs_bp=combined, header=header)
    print('Saved ' + save_path)


def leff_obs(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, ell_filename, pos_nl_path, she_nl_path, noise_ell_path,
             leff_path, save_dir):
    """
    Obtain a pseudo- cut-sky 'observation' by sampling from the Wishart distribution with effective l.

    Args:
        n_zbin (int): Number of redshift bins.
        pos_pos_dir (str): Path to directory containing input position-position power spectra.
        she_she_dir (str): Path to directory containing input shear-shear power spectra.
        pos_she_dir (str): Path to directory containing input position-shear power spectra.
        ell_filename (str): Filename of ells file within each input directory.
        pos_nl_path (str): Path to position noise power spectrum.
        she_nl_path (str): Path to shear noise power spectrum.
        noise_ell_path (str): Path to noise ells.
        leff_path (str): Path to leff map file.
        save_dir (str): Path to output directory.
    """

    n_fields = 2 * n_zbin

    # Load theory cls and convert to matrices
    theory_cls = like_w.load_cls(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir)
    theory_mats = like_w.cl_matrix(theory_cls, n_fields)

    # Load ells and do some consistency checks
    ell_pos_pos = np.loadtxt(os.path.join(pos_pos_dir, ell_filename))
    ell_she_she = np.loadtxt(os.path.join(she_she_dir, ell_filename))
    ell_pos_she = np.loadtxt(os.path.join(pos_she_dir, ell_filename))
    lmax = np.amax(ell_pos_pos)
    lmin = np.amin(ell_pos_she)
    n_ell = int(lmax - lmin + 1)
    assert np.allclose(ell_pos_pos, ell_she_she)
    assert np.allclose(ell_pos_pos, ell_pos_she)
    assert len(ell_pos_pos) == n_ell
    assert theory_mats.shape[0] == n_ell

    # Load noise cls and ells and trim to correct range
    pos_nl = np.loadtxt(pos_nl_path)
    she_nl = np.loadtxt(she_nl_path)
    noise_ell = np.loadtxt(noise_ell_path)
    noise_keep = np.logical_and(noise_ell >= lmin, noise_ell <= lmax)
    pos_nl = pos_nl[noise_keep]
    she_nl = she_nl[noise_keep]
    assert np.allclose(noise_ell[noise_keep], ell_pos_pos)

    # Convert noise cls to matrices (all diagonal)
    nl_nonzero = np.array([pos_nl, she_nl]*n_zbin)
    n_cls = int(n_fields * (n_fields + 1) / 2)
    nl_zero = np.zeros((n_cls - n_fields, n_ell))
    nl = np.concatenate((nl_nonzero, nl_zero))
    noise_mats = like_w.cl_matrix(nl, n_fields)

    # Load and apply leff map
    leff_map = np.loadtxt(leff_path)
    leff_l = leff_map[:, 0]
    leff_map = leff_map[(leff_l >= lmin) & (leff_l <= lmax)]
    assert np.allclose(ell_pos_pos, leff_map[:, 0])
    leffs = leff_map[:, 1]

    # For each l, sample from wishart with df = 2 * leff + 1 and scale = theory_cl_mat / df
    obs_mats = np.full_like(theory_mats, np.nan)
    zero = np.zeros(theory_mats.shape[1:])
    print(f'Starting at {time.strftime("%c")}')
    for i, (l, leff, theory_mat, noise_mat) in enumerate(zip(ell_pos_pos, leffs, theory_mats, noise_mats)):
        print(f'l = {l:.0f} / {lmax:.0f}', end='\r')

        df = 2 * leff + 1

        # If wishart isn't defined, set all 0 - no effect as will be excluded from likelihood anyway
        if df < n_fields:
            obs_mats[i] = zero
            continue

        scale = (theory_mat + noise_mat) * 1. / df
        obs = scipy.stats.wishart.rvs(df=df, scale=scale)
        assert is_pd(obs)
        obs_mats[i] = obs

    # Save to disk
    assert np.all(np.isfinite(obs_mats))
    obs_cls = like_w.cl_vector(obs_mats)
    save_cls_nob(obs_cls, n_zbin, save_dir, lmin=lmin)


def gaussian_samples(sim_bp_path, save_path):
    """
    Generate Gaussian random samples having the same mean and covariance as simulated bandpowers.
    Designed to use output from combine_sim_bp_batches.

    Args:
        sim_bp_path (str): Path to simulated bandpowers to match mean and covariance of.
        save_path (str): Path to save output to.
    """

    # Load data
    with np.load(sim_bp_path) as data:
        sim_bp = data['obs_bp']

    # Reshape to be a single vector per observation
    n_real, n_spec, n_bp = sim_bp.shape
    sim_bp = np.reshape(sim_bp, (n_real, n_spec * n_bp))

    # Calculate mean and covariance
    mean = np.mean(sim_bp, axis=0)
    cov = np.cov(sim_bp, rowvar=False)

    # Generate Gaussian samples
    gauss_bp = np.random.default_rng().multivariate_normal(mean, cov, size=n_real)

    # Reshape to the input shape
    gauss_bp = np.reshape(gauss_bp, (n_real, n_spec, n_bp))

    # Save to disk
    header = f'Output from {__file__}.gaussian_samples for input {sim_bp_path} at {time.strftime("%c")}'
    np.savez_compressed(save_path, gauss_bp=gauss_bp, header=header)
    print('Saved ' + save_path)


def sim_singlespec(input_cl_path, lmax, lmin_in, nl_path, mask_path, nside, n_loop, batch_size, lmin_out, save_dir):
    """
    Generate simple single-field full-spectrum simulations.

    Args:
        input_cl_path (str): Path to input power spectrum.
        lmax (int): Maximum l to simulate.
        lmin_in (int): l corresponding to first line in input power spectrum file.
        nl_path (str): Path to noise power spectrum.
        mask_path (str): Path to mask fits file.
        nside (int): Resolution of healpix maps to use.
        n_loop (int): Number of realisations to generate in total.
        batch_size (int): Number of realisations per batch, where one file is saved for each batch.
        lmin_out (int): Minimum l to save.
        save_dir (str): Path to output directory.
    """

    # Load input power spectrum
    input_cl = np.concatenate((np.zeros(lmin_in), np.loadtxt(input_cl_path, max_rows=(lmax - lmin_in + 1))))

    # Add noise
    nl = np.concatenate((np.zeros(lmin_in), np.loadtxt(nl_path, max_rows=(lmax - lmin_in + 1))))
    input_cl += nl

    # Load mask
    mask = hp.pixelfunc.ud_grade(hp.fitsfunc.read_map(mask_path, dtype=float, verbose=False), nside)

    n_batch = n_loop // batch_size
    for batch in range(n_batch):

        print(f'Starting batch {batch} at {time.strftime("%c")}')

        cl_fs = np.full((batch_size, lmax - lmin_out + 1), np.nan)
        cl_ma = cl_fs.copy()

        # For each realisation:
        for i in range(batch_size):

            print(f'Starting realisation {i} in batch {batch} at {time.strftime("%c")}')

            # Generate alms
            alm = hp.sphtfunc.synalm(input_cl, lmax=lmax, new=True)

            # Measure full-sky cls from alms
            cl_fs[i] = hp.sphtfunc.alm2cl(alm)[lmin_out:]

            # Transform to pixel space
            map_ = hp.sphtfunc.alm2map(alm, nside, verbose=False)

            # Apply mask
            map_ *= mask

            # Measure cut sky cls from map
            cl_ma[i] = hp.sphtfunc.anafast(map_, lmax=lmax)[lmin_out:]

        assert np.all(np.isfinite(cl_fs))
        assert np.all(np.isfinite(cl_ma))

        # Save batch
        header = f'output from {__file__}; batch {batch}'
        save_path = os.path.join(save_dir, f'batch_{batch}.npz')
        np.savez_compressed(save_path, cl_fs=cl_fs, cl_ma=cl_ma, header=header)
        print(f'Saved {save_path} at {time.strftime("%c")}')

    print(f'Done at {time.strftime("%c")}')


def combine_slics(input_filemask, tomo, save_filemask, lmax):
    """
    Combine all SLICS Cls for a particular tomographic bin into a single file, and remove l^2/2π prefactor.

    Args:
        input_filemask (str): Path to input SLICS files, with placeholders for {tomo} and {los}.
        tomo (int): SLICS tomographic bin index.
        save_filemask (str): Path to save combined files to, with placeholder for {tomo}.
        lmax (int): Maximum l to save.
    """

    # Identify input files
    input_files = glob.glob(input_filemask.format(tomo=tomo, los='[0-9]*'))
    n_file = len(input_files)

    cl = []
    for i, input_file in enumerate(input_files):
        print(f'Loading {i + 1} / {n_file}', end='\r')

        # Input is l^2 * Cl / (2π) so divide by this prefactor to obtain raw Cls
        ls_test, l2cl2pi = np.loadtxt(input_file, unpack=True)
        cls_ = 2 * np.pi * l2cl2pi / (ls_test ** 2)
        if i == 0:
            ls = ls_test
        else:
            assert np.allclose(ls, ls_test)
        cl.append(cls_[ls <= lmax])

    cl = np.stack(cl)
    save_path = save_filemask.format(tomo=tomo)
    np.savez_compressed(save_path, l=ls[ls <= lmax], cl=cl, tomo=tomo)
    print('Saved ' + save_path)


def noise_cls(gals_per_sq_arcmin, sigma_e, lmin, lmax, save_dir):
    """
    Save noise power spectra to file.

    Args:
        gals_per_sq_arcmin (float): Number of galaxies per square arcminute per redshift bin.
        sigma_e (float): Intrinsic galaxy ellipticity dispersion per component.
        lmin (int): Minimum l to save.
        lmax (int): Maximum l to save.
        save_dir (str): Path to output directory.
    """

    # Convert galaxies per square arcmin to per steradian
    n_i = gals_per_sq_arcmin * (60 * 180 / np.pi) ** 2

    # Calculate ell range, shear Nl = σ^2/Ni and position Nl = 1/Ni
    ell = np.arange(lmin, lmax + 1)
    pos_nl = (1 / n_i) * np.ones_like(ell)
    she_nl = (sigma_e ** 2 / n_i) * np.ones_like(ell)

    # Save ells
    ell_path = os.path.join(save_dir, 'noise_ell.txt')
    np.savetxt(ell_path, ell, header='ell')
    print('Saved ' + ell_path)

    # Save the two Nl spectra
    pos_nl_path = os.path.join(save_dir, 'pos_nl.txt')
    np.savetxt(pos_nl_path, pos_nl, header='pos_nl')
    print('Saved ' + pos_nl_path)
    she_nl_path = os.path.join(save_dir, 'she_nl.txt')
    np.savetxt(she_nl_path, she_nl, header='she_nl')
    print('Saved ' + she_nl_path)
