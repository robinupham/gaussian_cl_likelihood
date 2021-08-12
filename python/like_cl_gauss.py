"""
Likelihood module to evaluate the joint likelihood of a set of tomographic 3x2pt power spectra on the full sky
using a multivariate Gaussian likelihood.

The main functions are setup, which should be called once per analysis, and execute, which is called for every new
point in parameter space.
"""

import os.path
import numpy as np


def mvg_logpdf_fixedcov(x, mean, inv_cov):
    """
    Log-pdf of the multivariate Gaussian where the determinant and inverse of the covariance matrix are precomputed
    and fixed.
    Note that this neglects the additive constant: -0.5 * (len(x) * log(2 * pi) + log_det_cov), because it is
    irrelevant when comparing pdf values with a fixed covariance, but it means that this is not the normalised pdf.

    Args:
        x (1D numpy array): Vector value at which to evaluate the pdf.
        mean (1D numpy array): Mean vector of the multivariate Gaussian distribution.
        inv_cov (2D numpy array): Inverted covariance matrix.

    Returns:
        float: Log-pdf value.
    """

    dev = x - mean
    return -0.5 * (dev @ inv_cov @ dev)


def log_likelihood_single_l(theory_cl, noise_cl, obs_cl, inv_cov):
    """
    Returns the log-likelihood of a set of Cls for a single l.

    Args:
        theory_cl (1D numpy array): Set of theorys Cls for this l.
        noise_cl (1D numpy array): Set of noise Cls for this l.
        obs_cl (1D numpy array): Set of observed Cls for this l.
        inv_cov (2D numpy array): Inverted covariance matrix.

    Returns:
        float: Log-likelihood value.
    """

    mean = theory_cl + noise_cl
    return mvg_logpdf_fixedcov(obs_cl, mean, inv_cov)


def joint_log_likelihood(ell, theory_cl, noise_cl, obs_cl, inv_cov, lmax):
    """
    Returns the joint log-likelihood of a whole observed 3x2pt data vector.

    Args:
        ell (1D numpy array): Array of all ells.
        theory_cl (2D numpy array): Theory Cls, with ells along the first axis and different spectra along the second.
        noise_cl (2D numpy array): Noise Cls, in the same shape as theory_cl.
        obs_cl (2D numpy array): Observed Cls, in the same shape as theory_cl.
        inv_cov (2D numpy array): Inverted covariance matrix.
        lmax (int): Maximum ell to include in the likelihood.

    Returns:
        float: Joint log-likelihood value.
    """

    # Calculate lmin for contributions for consistency with Wishart,
    # which is only defined for 2l + 1 < x where x(x+1)/2 = n_spectra
    n_spectra = theory_cl.shape[1]
    nu_min = np.floor(np.sqrt(2 * n_spectra))
    lmin = np.ceil((nu_min - 1.) / 2)
    valid_l_range = ell[(ell >= lmin) & (ell <= lmax)]
    l_idx_offset = ell[0]

    # Evaluate log-likelihood for each ell separately and accumulate
    log_like = 0
    for l in valid_l_range:
        i = int(l - l_idx_offset)
        log_like += log_likelihood_single_l(theory_cl[i], noise_cl[i], obs_cl[i], inv_cov[i])

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


def matrix_indices(vector_idx, matrix_size):
    """
    Convert from vector index to matrix indices.

    Args:
        vector_idx (int): Vector index.
        matrix_size (int): Size along one axis of the square matrix to output indices for.

    Returns:
        (int, int): Row index, column index.
    """

    assert vector_idx < matrix_size * (matrix_size + 1) / 2, 'Invalid vector_idx for this matrix_size'

    # Work out which diagonal the element is on and its index on that diagonal, by iterating over the diagonals
    diag_length = matrix_size
    while vector_idx - diag_length >= 0:
        vector_idx -= diag_length
        diag_length -= 1
    diag = matrix_size - diag_length

    # Index at the top of the diagonal is (row = 0, col = diag),
    # so index of element is (row = vector_idx, col = diag + vector_idx)
    row = vector_idx
    col = diag + vector_idx
    return row, col


def vector_index(row_idx, col_idx, matrix_size):
    """
    Convert from matrix indices to vector index.

    Args:
        row_idx (int): Row index for matrix.
        col_idx (int): Column index for matrix.
        matrix_size (int): Size along one axis of the square matrix that input indices are for.

    Returns:
        int: Vector index corresponding to the input matrix indices.
    """

    # Only consider the upper triangle of the matrix by requiring that col >= row
    col = max(row_idx, col_idx)
    row = min(row_idx, col_idx)

    # Formula comes from standard sum over n
    diag = col - row
    return int(row + diag * (matrix_size - 0.5 * (diag - 1)))


def cl_cov(theory_cl, l, n_fields):
    """
    Returns the Gaussian covariance matrix of full-sky Cl estimates for a single l.

    Args:
        theory_cl (1D numpy array): All Cls for this l, in diagonal ordering.
        l (int): The l value.
        n_fields (int): The number of fields, where the number of spectra = n_fields * (n_fields + 1) / 2.

    Returns:
        2D numpy array: The covariance matrix for this l.
    """

    # Create empty covariance matrix
    n_spectra = len(theory_cl)
    cov_mat = np.full((n_spectra, n_spectra), np.nan)

    # Loop over all combinations of theory Cls
    for i, _ in enumerate(theory_cl):
        alpha, beta = matrix_indices(i, n_fields)

        for j, _ in enumerate(theory_cl):

            # Calculate all the relevant indices
            gamma, epsilon = matrix_indices(j, n_fields)
            alpha_gamma_idx = vector_index(alpha, gamma, n_fields)
            beta_epsilon_idx = vector_index(beta, epsilon, n_fields)
            alpha_epsilon_idx = vector_index(alpha, epsilon, n_fields)
            beta_gamma_idx = vector_index(beta, gamma, n_fields)

            # Calculate the covariance using the general Gaussian covariance equation
            # (see arXiv:2012.06267 eqn 6)
            cov = (theory_cl[alpha_gamma_idx] * theory_cl[beta_epsilon_idx]
                   + theory_cl[alpha_epsilon_idx] * theory_cl[beta_gamma_idx]) / (2 * l + 1.)
            cov_mat[i, j] = cov

    # Check for finite and symmetric - PD is not checked here because numerical issues mean that a valid set of Cls
    # can give a very slightly non-PD covariance matrix, but this doesn't affect the fixed-covariance results
    assert np.all(np.isfinite(cov_mat)), 'Covariance matrix not finite'
    assert np.allclose(cov_mat, cov_mat.T), 'Covariance matrix not symmetric'

    return cov_mat


def cl_invcov(ell, fid_cl, nl, n_field):
    """
    Calculate the inverse covariance matrix for all l (each l separately), from fiducial Cls including noise.

    Args:
        ell (1D numpy array): All l values, shape (n_ell,).
        fid_cl (2D numpy array): Theory Cls, shape (n_ell, n_spectra), with spectra in diagonal ordering.
        nl (2D numpy array): Noise Cls, shape (n_ell, n_spectra).
        n_field (int): Number of fields, such that n_spectra = n_field * (n_field + 1) / 2.

    Returns:
        inv_cov (3D numpy array): Inverse covariance for each ell, shape (n_ell, n_spectra, n_spectra).
    """

    # Some useful quantities
    n_ell = len(ell)
    n_spec = n_field * (n_field + 1) // 2

    # Calculate inverse covariance for each l
    inv_cov = np.full((n_ell, n_spec, n_spec), np.nan)
    for l_idx, l in enumerate(ell):
        cov = cl_cov(fid_cl[l_idx] + nl[l_idx], l, n_field)
        inv_cov[l_idx] = np.linalg.inv(cov)
    assert np.all(np.isfinite(inv_cov))

    return inv_cov


def setup(n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir, pos_nl_path, she_nl_path, noise_ell_path,
          fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax, leff_path=None):
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
        fid_pos_pos_dir (str): Path to the directory containing the fiducial theory position-position power spectra,
                               used for the covariance.
        fid_she_she_dir (str): Path to the directory containing the fiducial theory shear-shear power spectra.
        fid_pos_she_dir (str): Path to the directory containing the fiducial theory position-shear power spectra.
        lmax (int): Maximum l to use in the likelihood.
        leff_path (str, optional): Path to ell-ell_effective mapping, to replace each l with its corresponding l_eff
                                   when calculating the covariance.

    Returns:
        dict: Config dictionary to pass to execute.
    """

    # Calculate number of fields assuming 2 per redshift bin
    n_fields = 2 * n_zbin

    # Load obs Cls & ells
    obs_cl = load_cls(n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir)
    obs_ell_pos = np.loadtxt(os.path.join(obs_pos_pos_dir, 'ell.txt'))
    obs_ell_she = np.loadtxt(os.path.join(obs_she_she_dir, 'ell.txt'))
    obs_ell_shp = np.loadtxt(os.path.join(obs_pos_she_dir, 'ell.txt'))

    # Do some consistency checks within the obs Cls and ells
    assert np.allclose(obs_ell_pos, obs_ell_she)
    assert np.allclose(obs_ell_pos, obs_ell_shp)
    assert np.all([len(spec) == len(obs_ell_pos) for spec in obs_cl])

    # Load noise Cls & ells
    pos_nl = np.loadtxt(pos_nl_path)
    she_nl = np.loadtxt(she_nl_path)
    noise_ell = np.loadtxt(noise_ell_path)

    # Do some consistency checks within the noise Cls and ells
    assert len(noise_ell) == len(pos_nl)
    assert len(noise_ell) == len(she_nl)

    # Load fiducial Cls and ells
    fid_cl = load_cls(n_zbin, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir)
    fid_ell_pos = np.loadtxt(os.path.join(fid_pos_pos_dir, 'ell.txt'))
    fid_ell_she = np.loadtxt(os.path.join(fid_she_she_dir, 'ell.txt'))
    fid_ell_shp = np.loadtxt(os.path.join(fid_pos_she_dir, 'ell.txt'))

    # Do some consistency checks within the fiducial Cls and ells
    assert np.allclose(fid_ell_pos, fid_ell_she)
    assert np.allclose(fid_ell_pos, fid_ell_shp)
    assert np.all([len(spec) == len(fid_ell_pos) for spec in fid_cl])

    # Force consistent ell range between obs, noise and fiducial Cls
    lmin = np.amax((np.amin(obs_ell_pos), np.amin(noise_ell), np.amin(fid_ell_pos)))
    obs_ell_trimmed = obs_ell_pos[(obs_ell_pos >= lmin) & (obs_ell_pos <= lmax)]
    obs_cl = obs_cl[:, (obs_ell_pos >= lmin) & (obs_ell_pos <= lmax)]
    noise_ell_trimmed = noise_ell[(noise_ell >= lmin) & (noise_ell <= lmax)]
    pos_nl = pos_nl[(noise_ell >= lmin) & (noise_ell <= lmax)]
    she_nl = she_nl[(noise_ell >= lmin) & (noise_ell <= lmax)]
    fid_ell_trimmed = fid_ell_pos[(fid_ell_pos >= lmin) & (fid_ell_pos <= lmax)]
    fid_cl = fid_cl[:, (fid_ell_pos >= lmin) & (fid_ell_pos <= lmax)]
    assert np.allclose(obs_ell_trimmed, noise_ell_trimmed)
    assert np.allclose(obs_ell_trimmed, fid_ell_trimmed)

    # Calculate scaling factor used for numerical precision, then multiply everything by it
    scaling = 10 ** (2 + np.ceil(-np.log10(np.amax(fid_cl))))
    obs_cl *= scaling
    pos_nl *= scaling
    she_nl *= scaling
    fid_cl *= scaling

    # Convert obs and fiducial Cls into the required form
    obs_cl = obs_cl.T
    fid_cl = fid_cl.T

    # Form noise Cl vector, where noise is only added to auto-spectra
    nl_nonzero = np.array([pos_nl, she_nl]*n_zbin)
    n_cls = int(n_fields * (n_fields + 1) / 2)
    nl_zero = np.zeros((n_cls - n_fields, len(noise_ell_trimmed)))
    nl = np.concatenate((nl_nonzero, nl_zero)).T

    # Load leff map, trim to size and replace the ells for the covariance with their effective values
    if leff_path:
        leff_map = np.loadtxt(leff_path)
        leff_map = leff_map[(leff_map[:, 0] >= lmin) & (leff_map[:, 0] <= lmax)]
        assert np.allclose(leff_map[:, 0], obs_ell_trimmed)
        cov_l = leff_map[:, 1]
    else:
        cov_l = obs_ell_trimmed

    # Calculate inverse covariance matrix for fiducial Cls
    inv_cov = cl_invcov(cov_l, fid_cl, nl, n_fields)

    # Form config dictionary
    config = {
        'ell': obs_ell_trimmed,
        'obs_cl': obs_cl,
        'noise_cl': nl,
        'inv_cov': inv_cov,
        'lmax': lmax,
        'scaling': scaling
    }

    return config


def execute(theory_ell, theory_cl, config):
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
    ell = config['ell']
    obs_cl = config['obs_cl']
    noise_cl = config['noise_cl']
    inv_cov = config['inv_cov']
    lmin = np.amax((np.amin(theory_ell), np.amin(ell)))
    lmax = config['lmax']
    scaling = config['scaling']

    # Swap axes in theory Cl so it is indexed (l-lmin, spectrum)
    theory_cl = np.asarray(theory_cl).T

    # Force the two ell ranges to match (or throw error)
    lmin = np.amax((np.amin(theory_ell), np.amin(ell)))
    ell_trimmed = ell[ell >= lmin]
    if len(ell_trimmed) < len(ell):
        obs_cl = obs_cl[ell >= lmin]
        noise_cl = noise_cl[ell >= lmin]
    theory_ell_trimmed = theory_ell[(theory_ell >= lmin) & (theory_ell <= lmax)]
    theory_cl = theory_cl[(theory_ell >= lmin) & (theory_ell <= lmax)]
    assert np.allclose(ell_trimmed, theory_ell_trimmed)

    theory_cl *= scaling

    # Evaluate the likelihood
    return joint_log_likelihood(ell_trimmed, theory_cl, noise_cl, obs_cl, inv_cov, lmax)
