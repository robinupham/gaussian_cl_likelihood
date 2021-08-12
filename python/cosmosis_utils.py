"""
Contains utility functions for creating and post-processing CosmoSIS grids.
"""

import glob
import os.path
import re
import shutil
import tarfile
import warnings

import numpy as np


def combine_chain_output(input_dir, chain_subdir_mask='chain{i}', filemask='_{n}.tgz', clean=True):
    """
    Combine output from multiple chains into a single directory without clashing filenames.

    Args:
        input_dir (str): Path to directory containing all the chain subdirectories.
        chain_subdir_mask (str): Chain subdirectory name with chain number replaced with {i}.
        filemask (str): Filemask of files within chains, with number replaced with {n}.
        clean (bool, optional): Whether to delete chain subdirectories afterwards (default True).
    """

    chain_dir_mask = os.path.join(input_dir, chain_subdir_mask)

    # Loop over all chains (must be continuously numbered starting at 0)
    start_n = 0
    i = 0
    while os.path.isdir(chain_dir_mask.format(i=i)):

        # Loop over all files matching the filemask (mustn't contain exclamation marks)
        dir_name = chain_dir_mask.format(i=i)
        files = glob.glob(dir_name + filemask.format(n='*'))
        ns = [None]*len(files)
        for f_idx, f in enumerate(files):

            # Extract the current value of n, add start_n and copy to destination
            old_filename = f[len(dir_name):]
            start = filemask.format(n='!').index('!')
            stop = start + len(old_filename) - len(filemask.format(n=''))
            old_n = int(old_filename[start:stop])
            new_n = old_n + start_n
            ns[f_idx] = new_n
            new_path = os.path.join(input_dir, filemask.format(n=new_n))
            shutil.copyfile(f, new_path)
            print(f'Copied {new_n}', flush=True)

        # Delete directory if clean requested
        if clean:
            dir_name = chain_dir_mask.format(i=i)
            shutil.rmtree(dir_name)
            print(f'Source directory {dir_name} completed and deleted')

        start_n = max(ns) + 1

        i += 1
    print('Done             ')


def generate_chain_input(params, n_chains, output_dir):
    """
    Generate a custom number of chain text files for input to the list sampler in cosmosis, covering a whole regular
    N-dimensional grid.

    Args:
        params (dict): Dictionary containing one sub-dictionary for each parameter to be varied. Each subdirectory
                       should contain three items: min (minimum value), max (maximum value) and steps (number of steps
                       including endpoints) - see example below.
        n_chains (int): Number of chains to output.
        output_dir (str): Path to output directory where the text files will be saved.

    Example for params::

        params = {
            'cosmological_parameters--w': {
                'min': -1.5,
                'max': -0.5,
                'steps': 41
            },
            'cosmological_parameters--wa': {
                'min': -1.0,
                'max': 1.0,
                'steps': 51
            }
        }
    """

    # Calculate ranges of each individual parameter
    param_ranges = [np.linspace(params[param]['min'], params[param]['max'], params[param]['steps']) for param in params]

    # Combine into a flat grid of parameter values
    param_grid = np.stack(np.meshgrid(*param_ranges, indexing='ij'), axis=-1)
    flat_shape = (np.product([params[param]['steps'] for param in params]), len(params))
    param_list = np.reshape(param_grid, flat_shape)

    # Split into chains of near-equal size
    chains = np.array_split(param_list, n_chains)

    # Save chains to file
    header = ' '.join(params)
    for i, chain in enumerate(chains):
        chain_path = os.path.join(output_dir, f'chain{i}.txt')
        np.savetxt(chain_path, chain, header=header)
        print(f'Saved {chain_path}')


def extract_data(input_dir, filemask='_*.tgz', params_filename='cosmological_parameters/values.txt', nodelete=False,
                 target_filenames=None, nbin_3x2=None, nbin_shear=None):
    """
    Takes a directory full of CosmoSIS output (i.e. _0.tgz etc), extracts the desired output and throws away the rest.

    Either takes a predetermined list of target filenames, or automatically generates it for a tomographic 3x2pt or
    shear-only set of power spectra, depending on the values of target_filenames, nbin_3x2 and nbin_shear. Exactly one
    of these three must be set.

    Args:
        input_dir (str): Path to the directory containing the tar files.
        filemask (str, optional): Mask matching file names to extract. Defaults to '_*.tgz'.
        params_filename (str, optional): Path to the file containing cosmological parameters within each tar file.
                                         Defaults to 'cosmological_parameters/values.txt'.
        nodelete (bool, optional): Whether to retain the tar files after extraction (False) or delete them
                                   (True, default).
        target_filenames (list, optional): List of filenames to extract,
                                           e.g. ['shear_cl/ell.txt', 'shear_cl/bin_1_1.txt'].
        nbin_3x2 (int, None): Number of redshift bins to auto-generate target filenames for a 3x2pt analysis.
        nbin_shear (int, None): Number of redshift bins to auto-generate target filenames for a shear-only analysis.
    """

    assert target_filenames or nbin_3x2 or nbin_3x2
    if target_filenames is not None:
        assert nbin_3x2 is None and nbin_shear is None

    # Target filenames for tomographic 3x2pt analysis
    if nbin_3x2 is not None:
        assert target_filenames is None and nbin_shear is None
        target_filenames = ['galaxy_cl/ell.txt', 'shear_cl/ell.txt', 'galaxy_shear_cl/ell.txt']
        for bin1 in range(1, nbin_3x2 + 1):
            for bin2 in range(1, nbin_3x2 + 1):
                target_filenames.extend([f'galaxy_cl/bin_{bin1}_{bin2}.txt',
                                         f'shear_cl/bin_{bin1}_{bin2}.txt',
                                         f'galaxy_shear_cl/bin_{bin1}_{bin2}.txt'])
                if bin1 != bin2:
                    target_filenames.append(f'galaxy_shear_cl/bin_{bin2}_{bin1}.txt')

    # Target filenames for tomographic shear-only analysis
    if nbin_shear is not None:
        assert target_filenames is None and nbin_3x2 is None
        target_filenames = ['shear_cl/ell.txt']
        for bin1 in range(1, nbin_shear + 1):
            for bin2 in range(1, bin1 + 1):
                target_filenames.append(f'shear_cl/bin_{bin1}_{bin2}.txt')

    # Loop over all files matching the mask
    input_files = glob.glob(os.path.join(input_dir, filemask))
    if not input_files:
        warnings.warn('Nothing matching filemask in input directory')
    for i, input_file in enumerate(input_files):

        print(f'{i + 1} / {len(input_files)}')

        # Check it's some form of tar file
        if not tarfile.is_tarfile(input_file):
            warnings.warn(f'Skipping {input_file} which fits the filemask but is not a tar file')
            continue

        # Open tar and determine base path
        tar = tarfile.open(input_file)
        base_path = re.match(r'(.)*(/_)(\d)+(/)', tar.getmembers()[0].name)[0]

        for target_filename in target_filenames + [params_filename]:

            # Identify path within tar and extraction path
            old_name = base_path + target_filename
            new_name = os.path.join(os.path.abspath(input_file).replace('.tgz', '/'), target_filename)

            # Move to extraction path within tar, then extract
            target = tar.getmember(old_name)
            target.name = new_name
            tar.extract(new_name)

            # If nodelete, move back to original path inside tar
            if nodelete:
                target.name = old_name

        # Close and delete input file unless told not to
        tar.close()
        if not nodelete:
            os.remove(input_file)

    print('Done')
