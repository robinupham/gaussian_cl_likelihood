"""
Likelihood module to evaluate the joint likelihood of a set of tomographic 3x2pt power spectra on the full sky
using the exact Wishart likelihood.

The main functions are setup, which should be called once per analysis, and execute, which is called for every new
point in parameter space.
"""

import os.path
import numpy as np
import scipy.stats as stats


def cl_matrix(cls_, n_fields):
    """
    Convert a sequence of power spectra into a sequence of Cl matrices.

    Args:
        cls_ (2D numpy array): Cls to reshape, with input shape (n_spectra, n_ell).
        n_fields (int): Number of fields, such that n_spectra = n_fields * (n_fields + 1) / 2.

    Returns:
        3D numpy array: Cls with shape (n_ell, n_fields, n_fields).
    """

    cls_ = np.asarray(cls_)
    n_spectra = n_fields * (n_fields + 1) / 2.
    assert cls_.shape[0] == n_spectra, f'cls_.shape is {cls_.shape}; n_spectra is {n_spectra}'

    # Create output array
    n_ells = cls_.shape[1]
    cls_matrices = np.zeros((n_ells, n_fields, n_fields))

    # Rotate axes of input so it is now indexed (l index, spectrum index)
    cls_ = cls_.T

    # Fill matrices as appropriate
    for l_idx in range(n_ells):
        start_idx = 0
        for diag in range(n_fields):
            stop_idx = start_idx + n_fields - diag

            # Add the relevant diagonal, including to lower half for diag > 0
            cls_matrices[l_idx] += np.diag(cls_[l_idx, start_idx:stop_idx], k=diag)
            if diag > 0:
                cls_matrices[l_idx] += np.diag(cls_[l_idx, start_idx:stop_idx], k=-diag)

            start_idx = stop_idx

    return cls_matrices


def cl_vector(cl_matrices):
    """
    The inverse of cl_matrix: takes a sequence of Cl matrices and returns a vector of power spectra.

    Args:
        cl_matrices (3D numpy array): Cls to reshape into a vector, with input shape (n_ell, n_fields, n_fields).

    Returns:
        2D numpy array: Cls with shape (n_spectra, n_ell), with n_spectra = n_fields * (n_fields + 1) / 2.
    """

    # Check input and form output array
    n_ells, n_fields, n_fields_test = cl_matrices.shape
    assert n_fields == n_fields_test
    n_cls = int(n_fields * (n_fields + 1) / 2)
    res = np.full((n_cls, n_ells), np.nan)

    # Extract each matrix in turn
    for l_idx, cl_mat in enumerate(cl_matrices):
        start_idx = 0
        for diag in range(n_fields):
            stop_idx = start_idx + n_fields - diag
            res[start_idx:stop_idx, l_idx] = np.diag(cl_mat, k=diag)
            start_idx = stop_idx

    # Check every element has been filled
    assert np.all(np.isfinite(res))
    return res


def log_likelihood_single_l(l, theory_cl_matrix, noise_cl_matrix, obs_cl_matrix):
    """
    Returns the log-likelihood of a set of Cls for a single l according to the Wishart distribution.

    Args:
        l (int): Single l value.
        theory_cl_matrix (2D numpy array): Matrix of theory Cls for this l.
        noise_cl_matrix (2D numpy array): Matrix of noise Cls for this l.
        obs_cl_matrix (2D numpy array): Matrix of observed Cls for this l.

    Returns:
        float: Log-likelihood value.
    """

    # Wishart parameters
    nu = 2 * l + 1
    scale = (theory_cl_matrix + noise_cl_matrix) * 1. / nu

    # Wishart is only defined for df >= size of scale
    if nu < scale.shape[0]:
        return 0

    return stats.wishart.logpdf(obs_cl_matrix, df=nu, scale=scale)


def joint_log_likelihood(ells, theory_cl_matrices, noise_cl_matrices, obs_cl_matrices, lmax):
    """
    Return the joint log-likelihood of a whole observed 3x2pt data vector.

    Args:
        ells (1D numpy array): All l values.
        theory_cl_matrices (3D numpy array): Theory Cl matrices, shape (n_ell, n_spectra, n_spectra).
        noise_cl_matrices (3D numpy array): Noise Cl matrices, shape (n_ell, n_spectra, n_spectra).
        obs_cl_matrices (3D numpy array): Observed Cl matrices, shape (n_ell, n_spectra, n_spectra).
        lmax (int): Maximum l to include in the likelihood.

    Returns:
        float: Log-likelihood value.
    """

    log_like = 0
    for i, l in enumerate(ells[ells <= lmax]):
        log_like += log_likelihood_single_l(l, theory_cl_matrices[i], noise_cl_matrices[i], obs_cl_matrices[i])

    return log_like


def is_even(x):
    """
    True if x is even, false otherwise.

    Args:
        x (float): Number to test.

    Returns:
        bool: True if even.
    """
    return x % 2 == 0


def is_odd(x):
    """
    True if x is odd, false otherwise.

    Args:
        x (float): Number to test.

    Returns:
        bool: True if odd.
    """
    return x % 2 == 1


def load_cls(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, lmax=None, lmin=0):
    """
    Given the number of redshift bins and relevant directories, load power spectra (position, shear, cross) in the
    correct order (diagonal / healpy new=True ordering).
    If lmin is supplied, the output will be padded to begin at l=0.

    Args:
        n_zbin (int): Number of redshift bins.
        pos_pos_dir (str): Path to directory containing position-position power spectra.
        she_she_dir (str): Path to directory containing shear-shear power spectra.
        pos_she_dir (str): Path to directory containing position-shear power spectra.
        lmax (int, optional): Maximum l to load - if not supplied, will load all lines, which requires the individual
                              lmax of each file to be consistent.
        lmin (int, optional): Minimum l supplied. Output will be padded with zeros below this point.

    Returns:
        2D numpy array: All Cls, with different spectra along the first axis and increasing l along the second.
    """

    # Calculate number of fields assuming 1 position field and 1 shear field per redshift bin
    n_field = 2 * n_zbin

    # Load power spectra in 'diagonal order'
    spectra = []
    for diag in range(n_field):
        for row in range(n_field - diag):
            col = row + diag

            # Determine whether position-position, shear-shear or position-shear by whether the row and column are even,
            # odd or mixed
            if is_even(row) and is_even(col):
                cl_dir = pos_pos_dir
            elif is_odd(row) and is_odd(col):
                cl_dir = she_she_dir
            else:
                cl_dir = pos_she_dir

            # Extract the bins: for pos-pos and she-she the higher bin index goes first, for pos-she pos goes first
            bins = (row // 2 + 1, col // 2 + 1)
            if cl_dir in (pos_pos_dir, she_she_dir):
                bin1 = max(bins)
                bin2 = min(bins)
            else:
                if is_even(row): # even means pos
                    bin1, bin2 = bins
                else:
                    bin2, bin1 = bins

            cl_path = os.path.join(f'bin_{bin1}_{bin2}.txt')

            # Load with appropriate ell range
            max_rows = None if lmax is None else (lmax - lmin + 1)
            spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows)))
            spectra.append(spec)

    return np.asarray(spectra)


def setup(n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir, pos_nl_path, she_nl_path, noise_ell_path, lmax,
          leff_path=None):
    """
    Load and precompute everything that is fixed throughout parameter space. This should be called once per analysis,
    prior to any calls to execute.

    Args:
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

    Returns:
        dict: Config dictionary to pass to execute.
    """

    # Calculate number of fields assuming 2 per redshift bin
    n_fields = 2 * n_zbin

    # Load obs Cls & ells
    obs_cls = load_cls(n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir)
    obs_ell_pos = np.loadtxt(os.path.join(obs_pos_pos_dir, 'ell.txt'))
    obs_ell_she = np.loadtxt(os.path.join(obs_she_she_dir, 'ell.txt'))
    obs_ell_shp = np.loadtxt(os.path.join(obs_pos_she_dir, 'ell.txt'))

    # Do some consistency checks within the obs Cls and ells
    assert np.allclose(obs_ell_pos, obs_ell_she)
    assert np.allclose(obs_ell_pos, obs_ell_shp)
    assert np.all([len(spec) == len(obs_ell_pos) for spec in obs_cls])

    # Load noise Cls & ells
    pos_nl = np.loadtxt(pos_nl_path)
    she_nl = np.loadtxt(she_nl_path)
    noise_ell = np.loadtxt(noise_ell_path)

    # Do some consistency checks within the noise Cls and ells
    assert len(noise_ell) == len(pos_nl)
    assert len(noise_ell) == len(she_nl)

    # Force consistent ell range between obs and noise Cls
    lmin = np.amax((np.amin(obs_ell_pos), np.amin(noise_ell)))
    obs_ell_trimmed = obs_ell_pos[(obs_ell_pos >= lmin) & (obs_ell_pos <= lmax)]
    obs_cls = obs_cls[:, (obs_ell_pos >= lmin) & (obs_ell_pos <= lmax)]
    noise_ell_trimmed = noise_ell[(noise_ell >= lmin) & (noise_ell <= lmax)]
    pos_nl = pos_nl[(noise_ell >= lmin) & (noise_ell <= lmax)]
    she_nl = she_nl[(noise_ell >= lmin) & (noise_ell <= lmax)]
    assert np.allclose(obs_ell_trimmed, noise_ell_trimmed)

    # Convert obs Cls and noise Cls to matrices
    obs_cl_matrices = cl_matrix(obs_cls, n_fields)

    # Convert noise Cls to matrices (all diagonal)
    nl_nonzero = np.array([pos_nl, she_nl]*n_zbin)
    n_cls = int(n_fields * (n_fields + 1) / 2)
    nl_zero = np.zeros((n_cls - n_fields, len(noise_ell_trimmed)))
    nl = np.concatenate((nl_nonzero, nl_zero))
    noise_cl_matrices = cl_matrix(nl, n_fields)

    # If a path to a leff mapping is provided, load it to get leff
    if leff_path:
        leff_map = np.loadtxt(leff_path)
        leff_map = leff_map[(leff_map[:, 0] >= lmin) & (leff_map[:, 0] <= lmax)]
        assert np.allclose(leff_map[:, 0], obs_ell_trimmed)
        leff = leff_map[:, 1]
        leff_max = np.amax(leff) + 1e-5 # to avoid floating point error when comparing
    else:
        leff = None
        leff_max = None

    # Form config dictionary
    config = {
        'ells': obs_ell_trimmed,
        'obs_cl_matrices': obs_cl_matrices,
        'noise_cl_matrices': noise_cl_matrices,
        'lmax': lmax,
        'n_fields': n_fields,
        'leff': leff,
        'leff_max': leff_max
    }

    return config


def execute(theory_ells, theory_cls, config):
    """
    Perform some consistency checks then evaluate the likelihood for particular theory Cls.

    Args:
        theory_ell (1D numpy array): Ell range for all of the theory spectra (must be consistent between spectra).
        theory_cl (2D numpy array): Theory power spectra, in diagonal ordering, with shape (n_spectra, n_ell).
        config (dict): Config dictionary returned by setup.

    Returns:
        float: log-likelihood value.
    """

    # Pull fixed (model Cl-independent) parameters from config
    ells = config['ells']
    obs_cl_matrices = config['obs_cl_matrices']
    noise_cl_matrices = config['noise_cl_matrices']
    lmin = np.amax((np.amin(theory_ells), np.amin(ells)))
    lmax = config['lmax']
    n_fields = config['n_fields']
    leff = config['leff']
    leff_max = config['leff_max']

    # Convert theory Cls into matrices
    theory_cl_matrices = cl_matrix(theory_cls, n_fields)

    # Force the two ell ranges to match (or throw error)
    lmin = np.amax((np.amin(theory_ells), np.amin(ells)))
    ell_keep = ells >= lmin
    ells_trimmed = ells[ell_keep]
    if len(ells_trimmed) < len(ells):
        obs_cl_matrices = obs_cl_matrices[ell_keep]
        noise_cl_matrices = noise_cl_matrices[ell_keep]
    theory_ells_trimmed = theory_ells[(theory_ells >= lmin) & (theory_ells <= lmax)]
    theory_cl_matrices = theory_cl_matrices[(theory_ells >= lmin) & (theory_ells <= lmax)]
    assert np.allclose(ells_trimmed, theory_ells_trimmed)

    # Apply leff mapping
    if leff is not None:
        ells_trimmed = leff
        lmax = leff_max

    # Evaluate the likelihood
    return joint_log_likelihood(ells_trimmed, theory_cl_matrices, noise_cl_matrices, obs_cl_matrices, lmax)
