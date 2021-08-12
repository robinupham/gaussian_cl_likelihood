"""
Contains functions relating to calculating and plotting mutual information.
"""

import glob
import itertools
import time
import warnings

import matplotlib.patheffects
import matplotlib.pyplot as plt
import npeet.entropy_estimators
import numpy as np
import scipy.stats


def pit_stdnorm(data):
    """
    Apply a probability integral transform to map the marginals of data to a standard normal distribution.

    Args:
        data (2D numpy array): Bivariate input data with shape (2, n) where n is the number of samples and the first
                               axis indexes the two variables.
    Returns:
        2D numpy array: Input data with each variable mapped to a standard normal distribution. Shape (2, n'), where \
                        n' <= n as some samples may be removed if they map to non-finite values.
    """

    n = data.shape[1]
    x = data[0, :]
    y = data[1, :]

    # Calculate the CDF by ranking each element and then normalising
    x_rank = np.full(n, -1)
    x_rank[np.argsort(x)] = np.arange(n) + 1
    x_cdf = x_rank / np.amax(x_rank)
    y_rank = np.full(n, 1)
    y_rank[np.argsort(y)] = np.arange(n) + 1
    y_cdf = y_rank / np.amax(y_rank)

    # Convert to standard normal variables using the PPF
    stdnorm = scipy.stats.norm()
    data[0, :] = stdnorm.ppf(x_cdf)
    data[1, :] = stdnorm.ppf(y_cdf)

    # Remove non-finite realisations (boundaries)
    return data[:, np.all(np.isfinite(data), axis=0)]


def get_transcov(x, y, n_bins=100, cov_fac=None, check=True, pit=False):
    """
    Calculate additive transcovariance after whitening, using a histogram method.
    Implements equation 7 of Sellentin & Heavens 2018.

    Args:
        x (1D numpy array): Samples of the first variable.
        y (1D numpy array): Samples of the second variable.
        n_bins (int or (int, int), optional): Number of bins to use along each axis in the histogram, default 100.
        cov_fac (float, optional): Correction factor to apply to the inverse of an unbiased estimate of the data
                                   covariance to obtain an unbiased estimate of the inverse covariance, following
                                   Hartlap et al. (2007) eqn 17. If not supplied it is calculated.
        check (bool, optional): Check that the whitened data has the correct mean and covariance (default True).
        pit (bool, optional): Apply a probability integral transform to map the marginals to a standard normal
                              distribution (default False).

    Returns:
        float: Estimate of additive transcovariance of x and y after whitening.
    """

    assert x.shape == y.shape
    if cov_fac is None:
        n_real = x.shape[0]
        p = 2
        cov_fac = (n_real - p - 2.) / (n_real - 1.)
    data = np.stack((x, y))

    # Map marginals to standard normals if requested
    if pit:
        data = pit_stdnorm(data)

    # Apply the whitening matrix as the Cholesky decomposition of the bias-corrected inverse covariance matrix
    data -= np.mean(data, axis=-1)[:, np.newaxis]
    w = np.linalg.cholesky(cov_fac * np.linalg.inv(np.cov(data))).T
    data_white = w @ data
    if check:
        assert np.allclose(np.cov(data_white), np.identity(2), atol=1e-3)
        assert np.allclose(np.mean(data_white, axis=-1), 0)

    # Calculate observed histogram of sum
    sum_ = np.sum(data_white, axis=0)
    hist, edges = np.histogram(sum_, bins=n_bins, density=True)
    centres = edges[:-1] + 0.5 * (edges[1] - edges[0])

    # Calculate Gaussian histogram using the sample mean and variance of the summed whitened data, which in principle
    # should be 0 and 2 but in practice deviate from this slightly due to numerical imprecision
    gauss = scipy.stats.norm.pdf(centres, loc=np.mean(sum_), scale=np.std(sum_))

    # Calculate mean squared difference between the histograms
    return np.sum((hist - gauss) ** 2) / n_bins


def sim_bp_transcov(data_path_fs, data_path_ma, n_fields, n_bandpowers, save_path, pit=False):
    """
    Calculate whitened trancovariance of full- and cut-sky simulated bandpowers output by
    simulation.combine_sim_bp_batches.

    Args:
        data_path_fs (str): Path to full-sky simulated bandpowers.
        data_path_ma (str): Path to cut-sky simulated bandpowers.
        n_fields (int): Number of fields, such that n_spectra = n_fields * (n_fields + 1) / 2.
        n_bandpowers (int): Number of bandpowers.
        save_path (str): Path to save transcovariance matrices to.
        pit (bool, optional): If True, map the marginals to standard normal distributions before calculating
                              transcovariance (default False).
    """

    # Load the raw data and check shape
    with np.load(data_path_fs) as data:
        data_fs = data['obs_bp']
    with np.load(data_path_ma) as data:
        data_ma = data['obs_bp']
    n_spectra = n_fields * (n_fields + 1) // 2
    assert data_fs.shape[0] == data_ma.shape[0]
    assert data_fs.shape[1:] == (n_spectra, n_bandpowers)
    assert data_ma.shape[1:] == (n_spectra, n_bandpowers)

    # Calculate inverse covariance matrix debiasing factor for 2 parameters from Hartlap et al. 2007 eqn 17
    n_real = data_fs.shape[0]
    p = 2
    cov_fac = (n_real - p - 2.) / (n_real - 1.)

    # Semi-flatten so that shape is (n_real, n_spec * n_bp)
    len_data = n_spectra * n_bandpowers
    new_shape = (n_real, len_data)
    data_fs = np.reshape(data_fs, new_shape)
    data_ma = np.reshape(data_ma, new_shape)

    # Iterate twice over the second dimension to make a (n_spec * n_bp)x(n_spec * n_bp) matrix
    # (though this is symmetric so don't need to manually calculate all elements)
    transcov_mat_fs = np.full((len_data, len_data), np.nan)
    transcov_mat_ma = np.copy(transcov_mat_fs)
    count = 1
    total = len_data * (len_data - 1) // 2
    for i in range(len_data):
        transcov_mat_fs[i, i] = 0
        transcov_mat_ma[i, i] = 0
        for j in range(i):
            print(f'{count} / {total}')
            count += 1

            transcov_fs = get_transcov(data_fs[:, i], data_fs[:, j], cov_fac=cov_fac, pit=pit, check=False)
            transcov_ma = get_transcov(data_ma[:, i], data_ma[:, j], cov_fac=cov_fac, pit=pit, check=False)

            transcov_mat_fs[i, j] = transcov_fs
            transcov_mat_fs[j, i] = transcov_fs

            transcov_mat_ma[i, j] = transcov_ma
            transcov_mat_ma[j, i] = transcov_ma

    assert np.all(np.isfinite(transcov_mat_fs))
    assert np.all(np.isfinite(transcov_mat_ma))

    # Save result to file for post-processing
    header = (f'Output from {__file__}.sim_bp_transcov for input data_path_fs = {data_path_fs}, '
              f'data_path_ma = {data_path_ma}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, transcov_mat_fs=transcov_mat_fs, transcov_mat_ma=transcov_mat_ma, header=header)
    print('Saved ' + save_path)


def get_spectrum_table(n_fields):
    """
    Generate the lookup table for power spectra, which gives the two 'parent' auto-spectra for any cross-spectrum.

    Args:
        n_fields (int): Number of fields, such that the number of spectra = n_fields * (n_fields + 1) / 2.

    Returns:
        2D numpy array: Table of shape (n_cross_spectra, 3) where the three columns are: \
                        cross-spectrum index, parent 1 (row) index, parent 2 (column) index. Row and column refer to \
                        laying out all power spectra in the upper triangle of a (n_fields x n_fields) matrix.
    """

    # Initialise the table with values of -1
    n_cross_spectra = n_fields * (n_fields - 1) // 2
    spectrum_table = np.full((n_cross_spectra, 3), -1, dtype=int)

    # Fill the table following the standard diagonal ordering
    spectrum_idx = n_fields
    for diag in range(1, n_fields):
        for row in range(n_fields - diag):
            col = row + diag
            spectrum_table[spectrum_idx - n_fields] = (spectrum_idx, row, col)
            spectrum_idx += 1

    assert np.all(spectrum_idx > -1)
    return spectrum_table


def get_pair_idxs(spectra, bandpower, n_fields, n_bandpowers, spectrum_table=None, skip_dupe_check=False):
    """
    Return spectrum and bandpower indices of pairs of a given pair population.

    Possible values for spectra are:
        * same-auto              = same spectrum, auto-spectra only
        * same-cross             = same spectrum, cross-spectra only
        * auto-auto              = different auto-spectra
        * auto-cross-parent      = a cross-spectrum and one of its parent auto-spectra
        * auto-cross-nonparent   = a cross-spectrum and an auto-spectrum that is not its parent
        * cross-cross-sibling    = two cross-spectra sharing one parent auto-spectrum
        * cross-cross-nonsibling = two cross-spectra sharing no parent auto-spectra

    Possible values for bandpower are:
        * same
        * different (including duplicate pairs unless spectra is same-auto or same-cross, such that no two rows in the
          output table are complete duplicates)

    Args:
        spectra (str): Description of the relationship between spectra included in this pair population
                       (see list above).
        bandpower (str): Description of the relationship between bandpowers included in this pair population
                         (see list above).
        n_fields (int): Number of fields, such that the number of spectra = n_fields * (n_fields + 1) / 2.
        n_bandpowers (int): Number of bandpowers.
        spectrum_table (2D numpy array, optional): Precomputed spectrum lookup table produced by get_spectrum_table,
                                                   which is generated if not supplied.
        skip_dupe_check (bool, optional): If True, skip the check for duplicate rows in the output table
                                          (default False).

    Returns:
        list: List of 2D nested tuples giving the indices of all the pairs meeting the required description. Each \
              is: ((spectrum idx 1, spectrum idx 2), (bandpower idx 1, bandpower idx 2)).
    """

    # Build list of pairs of spectra, with no duplication, i.e. if (1, 2) is included then (2, 1) shouldn't be.
    if spectra == 'same-auto':
        spec_pairs = [(x, x) for x in range(n_fields)]
    elif spectra == 'same-cross':
        spec_pairs = [(x, x) for x in range(n_fields, n_fields * (n_fields + 1) // 2)]
    elif spectra == 'auto-auto':
        spec_pairs = [(y, x) for x in range(n_fields) for y in range(x)]
    else:
        if spectrum_table is None:
            spectrum_table = get_spectrum_table(n_fields)
        if spectra == 'auto-cross-parent':
            spec_pairs = list(itertools.chain.from_iterable(((x, y), (x, z)) for x, y, z in spectrum_table))
        elif spectra == 'auto-cross-nonparent':
            spec_pairs = [(x, a) for x, y, z in spectrum_table for a in range(n_fields) if a not in (y, z)]
        elif spectra == 'cross-cross-sibling':
            spec_pairs = [(x1, x2) for x1, y1, z1 in spectrum_table for x2, y2, z2 in spectrum_table
                          if x1 < x2 and (y1 in (y2, z2) or z1 in (y2, z2))]
        elif spectra == 'cross-cross-nonsibling':
            spec_pairs = [(x1, x2) for x1, y1, z1 in spectrum_table for x2, y2, z2 in spectrum_table
                          if x1 < x2 and y1 not in (y2, z2) and z1 not in (y2, z2)]
        else:
            raise ValueError(f'spectra string {spectra} not recognised')

    # Build list of pairs of bandpowers, including duplicate pairs except when spectra are same.
    if bandpower == 'same':
        bp_pairs = [(x, x) for x in range(n_bandpowers)]
    elif bandpower == 'different':
        if spectra in ('same-auto', 'same-cross'):
            bp_pairs = [(x, y) for x in range(n_bandpowers) for y in range(n_bandpowers) if x < y]
        else:
            bp_pairs = [(x, y) for x in range(n_bandpowers) for y in range(n_bandpowers) if x != y]
    else:
        raise ValueError(f'bandpower string {bandpower} not recognised')

    # Finally combine all combinations of spectra and bandpowers
    pairs = [((s1, s2), (b1, b2)) for s1, s2 in spec_pairs for b1, b2 in bp_pairs]

    # Check for duplicates and that the list has the expected size
    # (These can all fairly straightforward but a bit long to derive.)
    if not skip_dupe_check:
        assert not [1 for (s1, s2), (b1, b2) in pairs if ((s2, s1), (b2, b1)) in pairs]
    n_bp = n_bandpowers
    n_bpd = n_bp * (n_bp - 1) // 2        # Number of unique pairs of bandpowers
    n_as = n_fields                       # Number of auto-spectra
    n_cs = n_fields * (n_fields - 1) // 2 # Number of cross-spectra
    if spectra == 'same-auto' and bandpower == 'different':
        assert len(pairs) == n_bpd * n_as
    elif spectra == 'same-cross' and bandpower == 'different':
        assert len(pairs) == n_bpd * n_cs
    elif spectra == 'auto-auto' and bandpower == 'same':
        assert len(pairs) == n_bp * n_cs
    elif spectra == 'auto-auto' and bandpower == 'different':
        assert len(pairs) == 2 * n_bpd * n_cs
    elif spectra == 'auto-cross-parent' and bandpower == 'same':
        assert len(pairs) == 2 * n_bp * n_cs
    elif spectra == 'auto-cross-parent' and bandpower == 'different':
        assert len(pairs) == 4 * n_bpd * n_cs
    elif spectra == 'auto-cross-nonparent' and bandpower == 'same':
        assert len(pairs) == n_bp * (n_as - 2) * n_cs
    elif spectra == 'auto-cross-nonparent' and bandpower == 'different':
        assert len(pairs) == 2 * n_bpd * (n_as - 2) * n_cs
    elif spectra == 'cross-cross-sibling' and bandpower == 'same':
        assert len(pairs) == n_bp * (n_as - 2) * n_cs
    elif spectra == 'cross-cross-sibling' and bandpower == 'different':
        assert len(pairs) == 2 * n_bpd * (n_as - 2) * n_cs
    elif spectra == 'cross-cross-nonsibling' and bandpower == 'same':
        assert len(pairs) == n_bp * n_cs * (n_cs - 2 * (n_as - 2) - 1) // 2
    elif spectra == 'cross-cross-nonsibling' and bandpower == 'different':
        assert len(pairs) == n_bpd * n_cs * (n_cs - 2 * (n_as - 2) - 1)
    else:
        raise ValueError(f'Unexpected combination of spectra {spectra} and bandpower {bandpower}')

    return pairs


def calculate_all_mi(data_path_fs, data_path_ma, n_fields, n_bandpowers, save_path, data_label='obs_bp',
                     no_whiten=False, dilution_fac=1):
    """
    Calculate the mutual information (after whitening) for all pair populations within the full-sky and cut-sky
    simulated bandpowers output by simulation.combine_simbp_batches.
    Outputs one file per pair population.

    Args:
        data_path_fs (str): Path to full-sky simulated bandpowers.
        data_path_ma (str): Path to cut-sky simulated bandpowers.
        n_fields (int): Number of fields, such that the number of spectra = n_fields * (n_fields + 1) / 2.
        n_bandpowers (int): Number of bandpowers.
        save_path (str): Path to save output to, containing a placeholder {pop} which will be replaced with the
                         population label.
        data_label (str, optional): Label of the simulated data within the input npz files. Defaults to 'obs_bp', but
                                    for Gaussian samples output by simulation.gaussian_samples this should be replaced
                                    by 'gauss_bp'.
        no_whiten (bool, optional): If False, whitening is applied before mutual information estimation (default False).
        dilution_fac (float, optional): If > 1, the data sample will be randomly diluted by a factor of dilution_fac for
                                        faster but less accurate mutual information estimation. Default 1 (no dilution).
    """

    if no_whiten:
        warnings.warn('Whitening disabled')

    # Load the raw data and check shape
    with np.load(data_path_fs) as data:
        data_fs = data[data_label]
    with np.load(data_path_ma) as data:
        data_ma = data[data_label]
    n_spectra = n_fields * (n_fields + 1) // 2
    assert data_fs.shape[0] == data_ma.shape[0]
    assert data_fs.shape[1:] == (n_spectra, n_bandpowers)
    assert data_ma.shape[1:] == (n_spectra, n_bandpowers)

    # Calculate inverse covariance matrix debiasing factor for 2 parameters from Hartlap et al. (2007) eqn 17
    n_real = data_fs.shape[0]
    p = 2
    cov_fac = (n_real - p - 2.) / (n_real - 1.)

    populations = [
        ('same-auto', 'different'),
        ('same-cross', 'different'),
        ('auto-auto', 'same'),
        ('auto-auto', 'different'),
        ('auto-cross-parent', 'same'),
        ('auto-cross-parent', 'different'),
        ('auto-cross-nonparent', 'same'),
        ('auto-cross-nonparent', 'different'),
        ('cross-cross-sibling', 'same'),
        ('cross-cross-sibling', 'different'),
        ('cross-cross-nonsibling', 'same'),
        ('cross-cross-nonsibling', 'different'),
    ]

    # Precompute spectrum table
    spectrum_table = get_spectrum_table(n_fields)

    # Loop over populations
    for spectra, bandpower in populations:

        # Generate the population in terms of indices
        pop_idxs = get_pair_idxs(spectra, bandpower, n_fields, n_bandpowers, spectrum_table)
        pop_size = len(pop_idxs)
        if dilution_fac > 1:
            pop_size = len(pop_idxs) // dilution_fac
            pop_idxs = np.random.default_rng().choice(pop_idxs, size=pop_size, replace=False, axis=0)
        pop_mi_fs = np.full(pop_size, np.nan)
        pop_mi_ma = pop_mi_fs.copy()

        # For each pair in the list
        for i, (spec_idxs, bp_idxs) in enumerate(pop_idxs):
            print(f'Population {spectra}_{bandpower}: pair {i + 1} / {pop_size}')

            # Select out the actual data of the pair (tranposed to have 2 rows and n_real columns)
            pair_fs = data_fs[:, spec_idxs, bp_idxs].T
            pair_ma = data_ma[:, spec_idxs, bp_idxs].T

            if no_whiten:
                white_fs = pair_fs
                white_ma = pair_ma

            else:
                # Form each whitening matrix as the Cholesky decomposition of the bias-corrected
                # inverse covariance matrix
                w_fs = np.linalg.cholesky(cov_fac * np.linalg.inv(np.cov(pair_fs))).T
                w_ma = np.linalg.cholesky(cov_fac * np.linalg.inv(np.cov(pair_ma))).T

                # Whiten the data and check it's suitably white
                white_fs = w_fs @ pair_fs
                white_ma = w_ma @ pair_ma
                white_fs -= np.mean(white_fs, axis=1)[:, np.newaxis]
                white_ma -= np.mean(white_ma, axis=1)[:, np.newaxis]
                assert np.allclose(np.cov(white_fs), np.identity(2), atol=1e-3)
                assert np.allclose(np.cov(white_ma), np.identity(2), atol=1e-3)
                assert np.allclose(np.mean(white_fs, axis=1), 0)
                assert np.allclose(np.mean(white_ma, axis=1), 0)

            # Calculate and save MI of the whitened pair
            pop_mi_fs[i] = npeet.entropy_estimators.mi(white_fs[0], white_fs[1])
            pop_mi_ma[i] = npeet.entropy_estimators.mi(white_ma[0], white_ma[1])

        print()
        assert np.all(np.isfinite(pop_mi_fs)), pop_mi_fs
        assert np.all(np.isfinite(pop_mi_ma)), pop_mi_ma

        # Save to file
        dilution_str = f'_dil{dilution_fac}' if dilution_fac > 1 else ''
        whiten_str = '_nowhiten' if no_whiten else ''
        save_path = (save_path.format(pop=f'{spectra}_{bandpower}').replace('.npz', '')
                     + dilution_str + whiten_str + '.npz')
        header = (f'Output from {__file__}.calculate_all_mi for input: data_path_fs = {data_path_fs}, '
                  f'data_path_ma = {data_path_ma}, at {time.strftime("%c")}')
        np.savez_compressed(save_path, mi_fs=[pop_mi_fs], mi_ma=[pop_mi_ma], populations=[(spectra, bandpower)],
                            header=header) # lists are for backwards compatibility with previous code
        print('Saved ' + save_path)

    print('Done')


def combine_mi_files(input_filemask, save_path):
    """
    Combine single-population mutual information files output by calculation_all_mi to a single file.

    Args:
        input_filemask (str): glob-compatible filemask matching single-population files, e.g. 'mi_*.npz'.
        save_path (str): Path to save output to.
    """

    input_files = glob.glob(input_filemask)

    # Form lists of MI and populations to mimic old output of calculate_all_mi
    populations = []
    mi_fs = []
    mi_ma = []
    for i, input_file in enumerate(input_files):
        print(f'Loading input file {i + 1} / {len(input_files)}', end='\r')
        with np.load(input_file) as data:
            populations.append(tuple(np.squeeze(data['populations'])))
            mi_fs.append(np.squeeze(data['mi_fs']))
            mi_ma.append(np.squeeze(data['mi_ma']))

    # Convert to ragged object ndarrays to match original output
    mi_fs = np.array(mi_fs, dtype=object)
    mi_ma = np.array(mi_ma, dtype=object)

    # Save to a new combined file
    header = f'Output from {__file__}.combine_mi_files for input filemask {input_filemask} at {time.strftime("%c")}'
    np.savez_compressed(save_path, mi_fs=mi_fs, mi_ma=mi_ma, populations=populations, header=header)
    print('Saved ' + save_path)


def combine_mi_slics_gauss(input_filemask, save_path):
    """
    Alternative version of combine_mi_files for the output of calculate_mi_slics_gauss.
    Combine single-population mutual information files to a single file.

    Args:
        input_filemask (str): glob-compatible filemask matching single-population files, e.g. 'mi_*.npz'.
        save_path (str): Path to save output to.
    """

    input_files = glob.glob(input_filemask)

    # Form lists of MI and populations to mimic old output
    populations = []
    mi_slics = []
    mi_gauss = []
    for i, input_file in enumerate(input_files):
        print(f'Loading input file {i + 1} / {len(input_files)}', end='\r')
        with np.load(input_file) as data:
            populations.append(tuple(np.squeeze(data['populations'])))
            mi_slics.append(np.squeeze(data['mi_slics']))
            mi_gauss.append(np.squeeze(data['mi_gauss']))

    # Convert to ragged object ndarrays to match original output
    mi_slics = np.array(mi_slics, dtype=object)
    mi_gauss = np.array(mi_gauss, dtype=object)

    # Save to a new combined file
    header = f'Output from {__file__} for input filemask {input_filemask} at {time.strftime("%c")}'
    np.savez_compressed(save_path, mi_slics=mi_slics, mi_gauss=mi_gauss, populations=populations, header=header)
    print('Saved ' + save_path)


def get_pair_idxs_slics_gauss(spectra, bandpowers, n_fields, n_bandpowers):
    """
    Alternative version of get_pair_idxs for the SLICS vs Gaussian fields comparison.
    Return spectrum and bandpower indices of pairs of a given pair population.

    Possible values for spectra are:
        * same-spec
        * different-spec

    Possible values for bandpower are:
        * same-bp
        * different-bp (including duplicate pairs unless spectra is same-spec, such that no two rows in the output table
          are complete duplicates)

    Args:
        spectra (str): Description of the relationship between spectra included in this pair population
                       (see list above).
        bandpower (str): Description of the relationship between bandpowers included in this pair population
                         (see list above).
        n_fields (int): Number of fields, such that the number of spectra = n_fields * (n_fields + 1) / 2.
        n_bandpowers (int): Number of bandpowers.

    Returns:
        list: List of 2D nested tuples giving the indices of all the pairs meeting the required description. Each \
              is: ((spectrum idx 1, spectrum idx 2), (bandpower idx 1, bandpower idx 2)).
    """

    # Build list of pairs of spectra, with no duplication, i.e. if (1, 2) is included then (2, 1) shouldn't be.
    if spectra == 'same-spec':
        spec_pairs = [(x, x) for x in range(n_fields)]
    elif spectra == 'different-spec':
        spec_pairs = [(y, x) for x in range(n_fields) for y in range(x)]
    else:
        raise ValueError(f'spectra string {spectra} not recognised')

    # Build list of pairs of bandpowers, including duplicate pairs except when spectra are same.
    if bandpowers == 'same-bp':
        bp_pairs = [(x, x) for x in range(n_bandpowers)]
    elif bandpowers == 'different-bp':
        if spectra == 'same-spec':
            bp_pairs = [(x, y) for x in range(n_bandpowers) for y in range(n_bandpowers) if x < y]
        else:
            bp_pairs = [(x, y) for x in range(n_bandpowers) for y in range(n_bandpowers) if x != y]
    else:
        raise ValueError(f'bandpowers string {bandpowers} not recognised')

    # Finally combine all combinations of spectra and bandpowers
    pairs = [((s1, s2), (b1, b2)) for s1, s2 in spec_pairs for b1, b2 in bp_pairs]

    # Check that the list has the expected size
    n_bp = n_bandpowers
    n_bpd = n_bp * (n_bp - 1) // 2         # Number of unique pairs of bandpowers
    n_as = n_fields                        # Number of auto-spectra
    n_cs = n_fields * (n_fields - 1) // 2  # Number of cross-spectra (unique pairs of auto-spectra)
    if spectra == 'same-spec' and bandpowers == 'different-bp':
        assert len(pairs) == n_bpd * n_as
    elif spectra == 'different-spec' and bandpowers == 'same-bp':
        assert len(pairs) == n_bp * n_cs
    elif spectra == 'different-spec' and bandpowers == 'different-bp':
        assert len(pairs) == 2 * n_bpd * n_cs
    else:
        raise ValueError(f'Unexpected combination of spectra {spectra} and bandpowers {bandpowers}')

    return pairs


def calculate_mi_slics_gauss(slics_filemask, gauss_path, n_tomo, n_bp, n_real, save_path, no_whiten=False,
                             dilution_fac=1):
    """
    Calculate the mutual information (after whitening) for all pair populations within the SLICS and Gaussian field
    simulated bandpowers output by simulation.combine_slics and simulation.sim_flat.
    Outputs one file per pair population.

    Args:
        slics_filemask (str): Path to bandpowers from SLICS simulations,
                              with {tomo} as a placeholder for tomographic bin.
        gauss_path (str): Path to bandpowers from Gaussian fields simulations.
        n_tomo (int): Number of redshift bins.
        n_bp (int): Number of bandpowers.
        n_real (int): Number of realisations.
        save_path (str): Path to save output to, containing a placeholder {pop} which will be replaced with the
                         population label.
        no_whiten (bool, optional): If False, whitening is applied before mutual information estimation (default False).
        dilution_fac (float, optional): If > 1, the data sample will be randomly diluted by a factor of dilution_fac for
                                        faster but less accurate mutual information estimation. Default 1 (no dilution).
    """

    if no_whiten:
        warnings.warn('Whitening disabled')

    # Load raw data into a consistent shape
    bp_slics = np.full((n_tomo, n_bp, n_real), np.nan)
    leff_slics = np.full((n_tomo, n_bp), np.nan)
    for tomo in range(n_tomo):
        with np.load(slics_filemask.format(tomo=tomo)) as data:
            assert tomo == data['tomo']
            bp_slics[tomo] = data['cl'].T
            leff_slics[tomo] = data['l']
    assert np.all(np.isfinite(bp_slics))
    assert np.all(np.isfinite(leff_slics))
    assert np.allclose(np.diff(leff_slics, axis=0), 0)
    with np.load(gauss_path) as data:
        assert np.allclose(data['leff'], leff_slics[0])
        auto_indices = np.concatenate(([0], np.cumsum(np.arange(n_tomo, 1, -1))))
        bp_gauss = data['bps'][:, auto_indices, :]
    bp_gauss = np.moveaxis(bp_gauss, 0, -1)
    assert bp_slics.shape == bp_gauss.shape

    populations = [
        ('same-spec', 'different-bp'),
        ('different-spec', 'same-bp'),
        ('different-spec', 'different-bp')
    ]

    # For each population:
    for spectra, bandpowers in populations:

        # Generate the population in terms of indices
        pop_idxs = get_pair_idxs_slics_gauss(spectra, bandpowers, n_tomo, n_bp)
        pop_size = len(pop_idxs)
        if dilution_fac > 1:
            pop_size = len(pop_idxs) // dilution_fac
            pop_idxs = np.random.default_rng().choice(pop_idxs, size=pop_size, replace=False, axis=0)
        pop_mi_slics = np.full(pop_size, np.nan)
        pop_mi_gauss = pop_mi_slics.copy()

        # For each pair in the list
        for i, (spec_idxs, bp_idxs) in enumerate(pop_idxs):
            print(f'Population {spectra}_{bandpowers}: pair {i + 1} / {pop_size}')

            # Select out the actual data of the pair
            pair_slics = bp_slics[spec_idxs, bp_idxs, :]
            pair_gauss = bp_gauss[spec_idxs, bp_idxs, :]

            if no_whiten:
                white_slics = pair_slics
                white_gauss = pair_gauss

            else:
                # Form each whitening matrix as the Cholesky decomposition of the inverse covariance matrix
                w_slics = np.linalg.cholesky(np.linalg.inv(np.cov(pair_slics))).T
                w_gauss = np.linalg.cholesky(np.linalg.inv(np.cov(pair_gauss))).T

                # Whiten the data and check it's suitably white
                white_slics = w_slics @ pair_slics
                white_gauss = w_gauss @ pair_gauss
                white_slics -= np.mean(white_slics, axis=1)[:, np.newaxis]
                white_gauss -= np.mean(white_gauss, axis=1)[:, np.newaxis]
                assert np.allclose(np.cov(white_slics), np.identity(2))
                assert np.allclose(np.cov(white_gauss), np.identity(2))
                assert np.allclose(np.mean(white_slics, axis=1), 0)
                assert np.allclose(np.mean(white_gauss, axis=1), 0)

            # Calculate and save MI of the whitened pair
            pop_mi_slics[i] = npeet.entropy_estimators.mi(white_slics[0], white_slics[1])
            pop_mi_gauss[i] = npeet.entropy_estimators.mi(white_gauss[0], white_gauss[1])

        print()
        assert np.all(np.isfinite(pop_mi_slics))
        assert np.all(np.isfinite(pop_mi_gauss))

        # Save to file
        dilution_str = f'_dil{dilution_fac}' if dilution_fac > 1 else ''
        whiten_str = '_nowhiten' if no_whiten else ''
        save_path = (save_path.format(pop=f'{spectra}_{bandpowers}').replace('.npz', '')
                     + dilution_str + whiten_str + '.npz')
        header = (f'Output from {__file__} for input: slics_filemask = {slics_filemask}, gauss_path = {gauss_path}, '
                  f'at {time.strftime("%c")}')
        np.savez_compressed(save_path, mi_slics=[pop_mi_slics], mi_gauss=[pop_mi_gauss], # lists are for backwards
                            populations=[(spectra, bandpowers)], header=header)       # compatibility with previous code
        print('Saved ' + save_path)

    print('Done')


def plot_all_pairs(input_path, save_path=None):
    """
    Plot full-sky and cut-sky mutual information distributions for all populations, as output by combine_mi_files.

    Args:
        input_path (str): Path to MI estimates as output by combine_mi_files.
        save_path (str, optional): Path to save plot to, if supplied. If not supplied plot will be displayed.
    """

    # Load data in its original form
    with np.load(input_path, allow_pickle=True) as data:
        mi_fs = [y for y in data['mi_fs']]
        mi_ma = [y for y in data['mi_ma']]

    # Combine all pairs
    mi_fs = np.concatenate(mi_fs)
    mi_ma = np.concatenate(mi_ma)

    # Print means and medians
    print('Mean (full sky): ', np.mean(mi_fs))
    print('Mean (cut sky): ', np.mean(mi_ma))
    print('Median (full sky): ', np.median(mi_fs))
    print('Median (cut sky): ', np.median(mi_ma))

    # Tail counts
    print(np.count_nonzero(mi_fs > 0.02) / np.count_nonzero(mi_fs))
    print(np.count_nonzero(mi_ma > 0.02) / np.count_nonzero(mi_ma))

    # Plot histograms
    plt.rcParams.update({'font.size': 13})
    bins = 300
    edges = np.linspace(np.amin((mi_fs, mi_ma)), np.amax((mi_fs, mi_ma)), bins + 1)
    plt.hist(mi_fs, bins=edges, density=False, color='C0', alpha=.3, histtype='stepfilled', label='Full sky')
    plt.hist(mi_ma, bins=edges, density=False, color='C1', alpha=.3, histtype='stepfilled', label='Cut sky')

    # Annotation and legend
    annot = 'All pairs'
    plt.annotate(annot, (0.98, 0.97), xycoords='axes fraction', va='top', ha='right')
    plt.legend(frameon=False, loc='upper right', title='\n') # Easiest way to leave room for annotation

    # Axes labels
    plt.xlabel('Pairwise mutual information')
    plt.ylabel('Number of pairs of elements')

    plt.yscale('log')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_parents(input_path, save_path=None):
    """
    Plot full-sky and cut-sky mutual information distribution for auto-cross-parent pairs, with same bandpower and
    different bandpower pairs side-by-side.

    Args:
        input_path (str): Path to MI estimates as output by combine_mi_files.
        save_path (str, optional): Path to save plot to, if supplied. If not supplied plot will be displayed.
    """

    # Load data in its original form
    with np.load(input_path, allow_pickle=True) as data:
        populations = [tuple(pop) for pop in data['populations']]
        mi_fs = [y for y in data['mi_fs']]
        mi_ma = [y for y in data['mi_ma']]

    # Extract the ones we're interested in
    samebp_idx = populations.index(('auto-cross-parent', 'same'))
    diffbp_idx = populations.index(('auto-cross-parent', 'different'))
    mi_fs_samebp = mi_fs[samebp_idx]
    mi_ma_samebp = mi_ma[samebp_idx]
    mi_fs_diffbp = mi_fs[diffbp_idx]
    mi_ma_diffbp = mi_ma[diffbp_idx]

    # Plot histograms side-by-side
    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(ncols=2, figsize=plt.figaspect(1 / 3.))

    # Same BP
    bins = 50
    edges = np.linspace(np.amin((mi_fs_samebp, mi_ma_samebp)), np.amax((mi_fs_samebp, mi_ma_samebp)), bins + 1)
    ax[0].hist(mi_fs_samebp, bins=edges, density=False, color='C0', alpha=.3, histtype='stepfilled', label='Full sky')
    ax[0].hist(mi_ma_samebp, bins=edges, density=False, color='C1', alpha=.3, histtype='stepfilled', label='Cut sky')

    # Different BP
    bins = 50
    edges = np.linspace(np.amin((mi_fs_diffbp, mi_ma_diffbp)), np.amax((mi_fs_diffbp, mi_ma_diffbp)), bins + 1)
    ax[1].hist(mi_fs_diffbp, bins=edges, density=False, color='C0', alpha=.3, histtype='stepfilled', label='Full sky')
    ax[1].hist(mi_ma_diffbp, bins=edges, density=False, color='C1', alpha=.3, histtype='stepfilled', label='Cut sky')

    # Annotation, legend, axis labels
    bp_labels = ['Same bandpower', 'Different bandpower']
    for a, bp_label in zip(ax, bp_labels):
        annot = f'Auto\N{en dash}cross (parent)\n{bp_label}'
        a.annotate(annot, (0.98, 0.97), xycoords='axes fraction', va='top', ha='right')
        a.legend(frameon=False, loc='upper right', title='\n') # Easiest way to leave room for annotation
        a.set_xlabel('Pairwise mutual information')
        a.set_ylabel('Number of pairs of elements')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_matrix_slics_gauss(input_path, save_path=None):
    """
    Plot SLICS vs Gaussian fields mutual information matrix for same-spectrum pairs.

    Args:
        input_path (str): Path to mutual information estimates as output by combine_mi_slics_gauss.
        save_path (str, optional): Path to save plot to, if supplied. If not supplied plot will be displayed.
    """

    # Load data in its original form
    with np.load(input_path, allow_pickle=True) as data:
        populations = [tuple(pop) for pop in data['populations']]
        mi_slics = [y for y in data['mi_slics']]
        mi_gauss = [y for y in data['mi_gauss']]

    # Extract the same-spectrum population
    pop_idx = populations.index(('same-spec', 'different-bp'))
    mi_slics = mi_slics[pop_idx]
    mi_gauss = mi_gauss[pop_idx]
    assert mi_slics.shape == mi_gauss.shape

    # Identify spectra and bandpowers and form half matrices - exclude tomo0
    n_tomo = 4
    n_bp = 142
    pairs = get_pair_idxs_slics_gauss('same-spec', 'different-bp', n_tomo + 1, n_bp)
    assert len(pairs) == mi_slics.shape[0]
    mimat_slics = np.full((n_bp, n_bp, n_tomo), np.nan)
    mimat_gauss = mimat_slics.copy()
    for (pair_specs, pair_bps), pair_mi_slics, pair_mi_gauss in zip(pairs, mi_slics, mi_gauss):
        assert pair_specs[0] == pair_specs[1]
        tomo = pair_specs[0]
        if tomo == 0:
            continue
        bp1, bp2 = pair_bps
        assert bp1 != bp2
        mimat_slics[bp1, bp2, tomo - 1] = pair_mi_slics
        mimat_gauss[bp1, bp2, tomo - 1] = pair_mi_gauss
    mimat_slics = np.mean(mimat_slics, axis=-1)
    mimat_gauss = np.mean(mimat_gauss, axis=-1)

    # Combine into a single matrix - the resulting triangle has SLICS in the upper triangular
    # but this is plotted on the lower triangular because origin='lower'
    mimat_combined = np.asarray(np.triu(mimat_gauss)).T + np.triu(mimat_slics)

    # Plot matrix and colour bar
    plt.rcParams.update({'font.size': 13})
    plt.imshow(mimat_combined, origin='lower')
    cb = plt.colorbar()
    cb.set_label('Mutual information', rotation=270, labelpad=15)

    # Label the halves
    annot_params = {'xycoords':'axes fraction', 'c':'white', 'size':20, 'weight':'semibold'}
    annots = [plt.annotate('Gaussian fields', xy=(0.02, 0.97), va='top', **annot_params),
              plt.annotate('SLICS', xy=(0.98, 0.02), va='bottom', ha='right', **annot_params)]
    _ = [annot.set_path_effects([matplotlib.patheffects.withStroke(linewidth=1, foreground='k')]) for annot in annots]

    # Label axes and set ticks
    ticks = np.arange(141, step=20)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlabel('Bandpower')
    plt.ylabel('Bandpower')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_samespec_slics_gauss(input_path, save_path=None):
    """
    Plot all pairs and same-spec pairs side-by-side, for SLICS and Gaussian field bandpowers.

    Args:
        input_path (str): Path to mutual information estimates as output by combine_mi_slics_gauss.
        save_path (str, optional): Path to save plot to, if supplied. If not supplied plot will be displayed.
    """

    # Load data in its original form
    with np.load(input_path, allow_pickle=True) as data:
        populations = [tuple(pop) for pop in data['populations']]
        mi_slics = [y for y in data['mi_slics']]
        mi_gauss = [y for y in data['mi_gauss']]

    # Remove pairs including tomo0
    n_tomo = 5
    n_bp = 142
    keep = []
    for specs, bps in populations:
        pairs = get_pair_idxs_slics_gauss(specs, bps, n_tomo, n_bp)
        keep.append([0 not in spec_pairs for spec_pairs, _ in pairs])
    mi_slics_keep = []
    mi_gauss_keep = []
    for pop_keep, pop_mi_slics, pop_mi_gauss in zip(keep, mi_slics, mi_gauss):
        mi_slics_keep.append(pop_mi_slics[pop_keep])
        mi_gauss_keep.append(pop_mi_gauss[pop_keep])
    mi_slics = mi_slics_keep
    mi_gauss = mi_gauss_keep

    # Combine to get all pairs
    mi_slics_all = np.concatenate(mi_slics)
    mi_gauss_all = np.concatenate(mi_gauss)
    assert mi_slics_all.shape == mi_gauss_all.shape

    # Extract the same-spectrum population
    pop_idx = populations.index(('same-spec', 'different-bp'))
    mi_slics_samespec = mi_slics[pop_idx]
    mi_gauss_samespec = mi_gauss[pop_idx]
    assert mi_slics_samespec.shape == mi_gauss_samespec.shape

    plt.rcParams.update({'font.size': 13})
    _, axes = plt.subplots(ncols=2, figsize=plt.figaspect(1 / 3.))

    # Plot histograms
    bins = 150
    edges = np.linspace(np.amin((mi_slics_all, mi_gauss_all)), np.amax((mi_slics_all, mi_gauss_all)), bins + 1)
    axes[0].hist(mi_gauss_all, bins=edges, density=False, color='C0', alpha=.3, histtype='stepfilled',
                 label='Gaussian fields')
    axes[0].hist(mi_slics_all, bins=edges, density=False, color='C1', alpha=.3, histtype='stepfilled', label='SLICS')
    axes[0].set_yscale('log')

    bins = 150
    edges = np.linspace(np.amin((mi_slics_samespec, mi_gauss_samespec)),
                        np.amax((mi_slics_samespec, mi_gauss_samespec)), bins + 1)
    axes[1].hist(mi_gauss_samespec, bins=edges, density=False, color='C0', alpha=.3, histtype='stepfilled',
                 label='Gaussian fields')
    axes[1].hist(mi_slics_samespec, bins=edges, density=False, color='C1', alpha=.3, histtype='stepfilled',
                 label='SLICS')
    axes[1].set_yscale('log')

    # Annotation, legend, axis labels
    annots = ['All pairs', 'Same spectrum\nDifferent bandpower']
    for ax, annot in zip(axes, annots):
        ax.annotate(annot, (0.98, 0.97), xycoords='axes fraction', va='top', ha='right')
        ax.legend(frameon=False, loc='upper left')
        ax.set_xlabel('Pairwise mutual information')
        ax.set_ylabel('Number of pairs of elements')
        ax.set_ylim(top=10**(1.1 * np.log10(ax.get_ylim()[1])))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_transcov(input_path, save_path=None):
    """
    Plot full-sky and cut-sky distributions of transcovariance, as output by sim_bp_transcov.

    Args:
        input_path (str): Path to transcovariance estimates produced by sim_bp_transcov.
        save_path (str, optional): Path to save plot to, if supplied. If not supplied plot will be displayed.
    """

    # Load data
    with np.load(input_path) as data:
        transcov_mat_fs = data['transcov_mat_fs']
        transcov_mat_ma = data['transcov_mat_ma']

    # Extract relevant components (i.e. exclude duplicates and the diagonal)
    n_el = transcov_mat_fs.shape[0]
    idx = np.triu_indices(n_el, k=1)
    tc_fs = transcov_mat_fs[idx]
    tc_ma = transcov_mat_ma[idx]

    # Plot histograms
    plt.rcParams.update({'font.size': 13})
    bins = 300
    edges = np.linspace(np.amin((tc_fs, tc_ma)), np.amax((tc_fs, tc_ma)), bins + 1)
    plt.hist(tc_fs, bins=edges, density=False, color='C0', alpha=.3, histtype='stepfilled', label='Full sky')
    plt.hist(tc_ma, bins=edges, density=False, color='C1', alpha=.3, histtype='stepfilled', label='Cut sky')

    plt.xlabel(r'Transcovariance $S^{\!\!+}$')
    plt.ylabel('Number of pairs of elements')

    plt.yscale('log')

    # Annotation and legend
    annot = 'All pairs'
    plt.annotate(annot, (0.98, 0.97), xycoords='axes fraction', va='top', ha='right')
    plt.legend(frameon=False, title='\n')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_sim_gauss(sim_path, gauss_path, save_path=None):
    """
    Plot mutual information distributions for simulations and Gaussian samples side-by-side.

    Args:
        sim_path (str): Path to mutual information estimates from simulated bandpowers as output by combine_mi_files.
        gauss_path (str): Path to mutual information estimates from Gaussian samples as output by combine_mi_files.
        save_path (str, optional): Path to save plot to, if supplied. If not supplied plot will be displayed.
    """

    # Load data in its original form
    with np.load(sim_path, allow_pickle=True) as data:
        mi_sim_fs = [x for x in data['mi_fs']]
        mi_sim_ma = [x for x in data['mi_ma']]
    with np.load(gauss_path, allow_pickle=True) as data:
        mi_gauss_fs = [x for x in data['mi_fs']]
        mi_gauss_ma = [x for x in data['mi_ma']]

    # Combine all pairs
    mi_sim_fs = np.concatenate(mi_sim_fs)
    mi_sim_ma = np.concatenate(mi_sim_ma)
    mi_gauss_fs = np.concatenate(mi_gauss_fs)
    mi_gauss_ma = np.concatenate(mi_gauss_ma)

    # Plot histograms
    bins = 300
    edges = np.linspace(-2.45e-2, 0.11, bins + 1)
    plt.rcParams.update({'font.size': 13})
    _, ax = plt.subplots(ncols=2, figsize=plt.figaspect(1 / 3.))
    plt.subplots_adjust(wspace=.3)

    ax[0].hist(mi_sim_fs, bins=edges, density=False, color='C0', alpha=.3, histtype='stepfilled', label='Simulations')
    ax[0].hist(mi_gauss_fs, bins=edges, density=False, color='C2', alpha=.3, histtype='stepfilled',
               label='Gaussian samples')

    ax[1].hist(mi_sim_ma, bins=edges, density=False, color='C1', alpha=.3, histtype='stepfilled', label='Simulations')
    ax[1].hist(mi_gauss_ma, bins=edges, density=False, color='C2', alpha=.3, histtype='stepfilled',
               label='Gaussian samples')

    # Annotation and legend
    ax[0].annotate('All pairs, full sky', (0.98, 0.97), xycoords='axes fraction', va='top', ha='right')
    ax[1].annotate('All pairs, cut sky', (0.98, 0.97), xycoords='axes fraction', va='top', ha='right')

    for a in ax:
        a.set_xlabel('Pairwise mutual information')
        a.set_ylabel('Number of pairs of elements')
        a.set_xlim(-2.45e-2, 0.11)
        a.set_yscale('log')
        a.legend(frameon=False, loc='upper right', title='\n') # Easiest way to leave room for annotation

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_vs_l(input_path, pbl_path, lmin, save_path=None):
    """
    Plot full-sky and cut-sky mutual information vs l for all same-bandpower pairs.

    Args:
        input_path (str): Path to mutual information estimates as output by combine_mi_files.
        pbl_path (str): Path to bandpower binning matrix.
        lmin (int): l corresponding to lower bound of first bandpower.
        save_path (str, optional): Path to save plot to, if supplied. If not supplied plot will be displayed.
    """

    # Load data
    with np.load(input_path, allow_pickle=True) as data:
        populations = [tuple(pop) for pop in data['populations']]
        mi_fs = [x for x in data['mi_fs']]
        mi_ma = [x for x in data['mi_ma']]

    # Keep only same-bandpower data
    mi_fs_keep = []
    mi_ma_keep = []
    populations_keep = []
    for (spectra, bps), pop_mi_fs, pop_mi_ma in zip(populations, mi_fs, mi_ma):
        if bps == 'same':
            mi_fs_keep.append(pop_mi_fs)
            mi_ma_keep.append(pop_mi_ma)
            populations_keep.append((spectra, bps))
    mi_fs = mi_fs_keep
    mi_ma = mi_ma_keep
    populations = populations_keep

    # Sort MI into bandpowers
    n_fields = 10
    n_bps = 10
    mi_fs_perbp = [[] for _ in range(n_bps)]
    mi_ma_perbp = [[] for _ in range(n_bps)]
    for (spectra, bandpowers), pop_mi_fs, pop_mi_ma in zip(populations_keep, mi_fs, mi_ma):
        pairs = get_pair_idxs(spectra, bandpowers, n_fields, n_bps, skip_dupe_check=True)
        for (_, bps), this_mi_fs, this_mi_ma in zip(pairs, pop_mi_fs, pop_mi_ma):
            this_bp = bps[0]
            assert bps == (this_bp, this_bp)
            mi_fs_perbp[this_bp].append(this_mi_fs)
            mi_ma_perbp[this_bp].append(this_mi_ma)

    # Load the Pbl matrix and use it to calculate the bin centres for the bandpowers
    pbl = np.loadtxt(pbl_path)
    n_ell = pbl.shape[1]
    band_edges = [np.amin(np.nonzero(band)[0]) + lmin for band in pbl] + [n_ell + lmin - 1]
    band_centres = band_edges[:-1] + .5 * np.diff(band_edges)

    # Calculate means and standard deviations
    mean_mi_fs_perbp = np.array([np.mean(bp_mi) for bp_mi in mi_fs_perbp])
    mean_mi_ma_perbp = np.array([np.mean(bp_mi) for bp_mi in mi_ma_perbp])
    std_mi_fs_perbp = np.array([np.std(bp_mi) for bp_mi in mi_fs_perbp])
    std_mi_ma_perbp = np.array([np.std(bp_mi) for bp_mi in mi_ma_perbp])
    hi_fs = mean_mi_fs_perbp + .5 * std_mi_fs_perbp
    lo_fs = mean_mi_fs_perbp - .5 * std_mi_fs_perbp
    hi_ma = mean_mi_ma_perbp + .5 * std_mi_ma_perbp
    lo_ma = mean_mi_ma_perbp - .5 * std_mi_ma_perbp

    # Plot mean and shade standard deviation
    plt.rcParams.update({'font.size': 13})
    plt.plot(band_centres, mean_mi_fs_perbp, label='Full sky', lw=3, c='C0')
    plt.plot(band_centres, mean_mi_ma_perbp, label='Cut sky', lw=3, c='C1', ls='--')
    plt.fill_between(band_centres, lo_fs, hi_fs, color='C0', alpha=.3)
    plt.fill_between(band_centres, lo_ma, hi_ma, color='C1', alpha=.3)

    # Title and legend
    annot = 'Same-bandpower pairs\nMean & standard deviation'
    plt.annotate(annot, (0.98, 0.97), xycoords='axes fraction', va='top', ha='right')
    plt.legend(handlelength=4, frameon=False, title='\n\n')

    plt.xlabel(r'$\ell$')
    plt.ylabel('Pairwise mutual information')
    plt.xscale('log')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()


def plot_whitening(nowhiten_path, whiten_path, n_fields, n_bandpowers, save_path=None):
    """
    Plot full-sky and cut-sky mutual information distributions for adjacent bandpowers, with and without whitening.

    Args:
        nowhiten_path (str): Path to mutual information estimates without whitening, as output by combine_mi_files.
        whiten_path (str): Path to mutual information estimates with whitening, as output by combine_mi_files.
        n_fields (int): Number of fields, such that the number of power spectra = n_fields * (n_fields + 1) / 2.
        n_bandpowers (int): Number of bandpowers.
        save_path (str, optional): Path to save plot to, if supplied. If not supplied plot will be displayed.
    """

    # Load same-spectrum data
    with np.load(nowhiten_path, allow_pickle=True) as data:
        for pop, mi_fs_i, mi_ma_i in zip(data['populations'], data['mi_fs'], data['mi_ma']):
            if tuple(pop) == ('same-auto', 'different'):
                mi_fs_auto_nowhite = np.squeeze(mi_fs_i)
                mi_ma_auto_nowhite = np.squeeze(mi_ma_i)
            elif tuple(pop) == ('same-cross', 'different'):
                mi_fs_cross_nowhite = np.squeeze(mi_fs_i)
                mi_ma_cross_nowhite = np.squeeze(mi_ma_i)
    with np.load(whiten_path, allow_pickle=True) as data:
        for pop, mi_fs_i, mi_ma_i in zip(data['populations'], data['mi_fs'], data['mi_ma']):
            if tuple(pop) == ('same-auto', 'different'):
                mi_fs_auto_white = np.squeeze(mi_fs_i)
                mi_ma_auto_white = np.squeeze(mi_ma_i)
            elif tuple(pop) == ('same-cross', 'different'):
                mi_fs_cross_white = np.squeeze(mi_fs_i)
                mi_ma_cross_white = np.squeeze(mi_ma_i)

    # Generate pairs of indices to identify MI data
    pairs_auto = get_pair_idxs('same-auto', 'different', n_fields, n_bandpowers)
    pairs_cross = get_pair_idxs('same-cross', 'different', n_fields, n_bandpowers)

    # Extract MI for adjacent bandpowers
    mi_fs_nowhite = []
    mi_ma_nowhite = []
    mi_fs_white = []
    mi_ma_white = []
    for (_, bps), mi_fs_pair, mi_ma_pair in zip(pairs_auto, mi_fs_auto_nowhite, mi_ma_auto_nowhite):
        if np.abs(np.diff(bps)) == 1:
            mi_fs_nowhite.append(mi_fs_pair)
            mi_ma_nowhite.append(mi_ma_pair)
    for (_, bps), mi_fs_pair, mi_ma_pair in zip(pairs_cross, mi_fs_cross_nowhite, mi_ma_cross_nowhite):
        if np.abs(np.diff(bps)) == 1:
            mi_fs_nowhite.append(mi_fs_pair)
            mi_ma_nowhite.append(mi_ma_pair)
    for (_, bps), mi_fs_pair, mi_ma_pair in zip(pairs_auto, mi_fs_auto_white, mi_ma_auto_white):
        if np.abs(np.diff(bps)) == 1:
            mi_fs_white.append(mi_fs_pair)
            mi_ma_white.append(mi_ma_pair)
    for (_, bps), mi_fs_pair, mi_ma_pair in zip(pairs_cross, mi_fs_cross_white, mi_ma_cross_white):
        if np.abs(np.diff(bps)) == 1:
            mi_fs_white.append(mi_fs_pair)
            mi_ma_white.append(mi_ma_pair)

    # Plot histograms
    plt.rcParams.update({'font.size': 13})
    _, axes = plt.subplots(ncols=2, figsize=plt.figaspect(1 / 3.))
    bins = 50
    for ax, tohist_fs, tohist_ma in zip(axes, [mi_fs_nowhite, mi_fs_white], [mi_ma_nowhite, mi_ma_white]):
        edges = np.linspace(np.amin((tohist_fs, tohist_ma)), np.amax((tohist_fs, tohist_ma)), bins + 1)
        ax.hist(tohist_fs, bins=edges, density=False, color='C0', alpha=.3, histtype='stepfilled', label='Full sky')
        ax.hist(tohist_ma, bins=edges, density=False, color='C1', alpha=.3, histtype='stepfilled', label='Cut sky')
        ax.set_xlabel('Pairwise mutual information')
        ax.set_ylabel('Number of pairs of elements')

    # Annotation and legend
    for ax, label in zip(axes, ['Unwhitened', 'Whitened']):
        annot = 'Adjacent bandpowers\n' + label
        ax.annotate(annot, (0.98, 0.97), xycoords='axes fraction', va='top', ha='right')
        ax.legend(frameon=False, loc='upper right', title='\n') # Easiest way to leave room for annotation

    axes[1].set_xticks([-0.01, 0, 0.01, 0.02])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved ' + save_path)
    else:
        plt.show()
    plt.clf()
