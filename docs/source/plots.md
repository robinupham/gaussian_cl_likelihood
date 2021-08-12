Steps to produce all plots
==========================

Below are the steps to produce all plots in [Upham, Brown & Whittaker 2021, "Sufficiency of a Gaussian power spectrum likelihood for accurate cosmology from upcoming weak lensing surveys", arXiv:2012.06267](https://arxiv.org/abs/2012.06267).

## Figure 1: Histograms of Wishart and Gaussian 1D posterior maxima and per-realisation difference

a) Produce 1D theory grid of w0 (note that these instructions are general for any N-D grid):

1. Generate N-D chains with `cosmosis_utils.generate_chain_input`:

    ```python
    [python]

    params = {
        'cosmological_parameters--w': {
            'min': -1.06,
            'max': -0.94,
            'steps': 99
        }
    }
    n_chains = 11 # for a 12-CPU machine, to leave one free
    output_dir = 'path-to-output-dir'

    cosmosis_utils.generate_chain_input(params, n_chains, output_dir)
    ```

2. Set input and output paths in `tomo_3x2_pipeline.ini`.

3. Set chains off running in parallel with `run_cosmosis_chains.sh':

    ```bash
    [bash]

    bash run_cosmosis_chains.sh
    ```

4. Once finished, combine chain output with `cosmosis_utils.combine_chain_output`:

    ```python
    [python]

    input_dir = 'path-to-cosmosis-output-directory' chain_subdir_mask = 'chain{i}'
    filemask = '_{n}.tgz'
    clean = True

    cosmosis_utils.combine_chain_output(input_dir, chain_subdir_mask, filemask, clean)
    ```

5. Extract required files from .tgz archives using `cosmosis_utils.extract_data`:

    ```python
    [python]

    input_dir = 'path-to-cosmosis-output-directory'
    filemask = '_*.tgz'
    params_filename = 'cosmological_parameters/values.txt'
    nodelete = False # or True if you want to retain the .tgz files in case you want to extract anything else later
    nbin_3x2 = 5

    extract_data(input_dir, filemask=filemask, params_filename=params_filename, nodelete=nodelete, nbin_3x2=nbin_3x2)
    ```

b) Produce noise power spectra using `simulation.noise_cls`:

```python
[python]

n_zbin = 5
gals_per_sq_arcmin = 30 / n_zbin
sigma_e = 0.3
lmin = 2
lmax = 2000
save_dir = 'directory-to-save-noise-cls-to'

noise_cls(gals_per_sq_arcmin, sigma_e, lmin, lmax, save_dir)
```

c) Generate repeated full-sky observations of all Cls using `simulation.sim_cls_fullsky`:

```python
[python]

n_zbin = 5
fiducial_cls_dir = 'path-to-fiducial-Cls-directory' # This can be one of the sub-directories in the grid produced in step (a).
pos_pos_in_dir = fiducial_cls_dir + '/galaxy_cl'
she_she_in_dir = fiducial_cls_dir + '/shear_cl'
she_she_in_dir = fiducial_cls_dir + '/galaxy_shear_cl'
lmax = 2000
lmin_in = 2
pos_nl_path = 'path-to-position-noise-cls'
she_nl_path = 'path-to-shear-noise-cls'
lmin_out = 2
n_loop = 27000
batch_size = 1000
save_dir = 'directory-to-save-batches-into'

sim_cls_fullsky(n_zbin, pos_pos_in_dir, she_she_in_dir, pos_she_in_dir, lmax, lmin_in, pos_nl_path, she_nl_path,
                    lmin_out, n_loop, batch_size, save_dir)
```

d) Loop over observations, run the Gaussian and Wishart likelihoods for each and determine the maximum, using `run_likelihoods.max_like_1d`:

```python
[python]

obs_filemask = 'path-to-output-from-above/[0-9]*.npz'
theory_filemask = 'path-to-grid-directory/_[0-9]*/'
varied_param = 'w'
save_dir = 'directory-to-save-results-to'
batch_size = 1000
n_zbin = 5
fid_dir = 'path-to-fiducial-Cls-directory'
pos_subdir = 'galaxy_cl'
she_subdir = 'shear_cl'
pos_she_subdir = 'galaxy_shear_cl'
ell_filename = 'ell.txt'
pos_nl_path = 'path-to-pos_nl'
she_nl_path = 'path-to-she_nl'
noise_ell_path = 'path-to-noise-ell'
lmax = 2000
lmin = 2

max_like_1d(theory_filemask, obs_filemask, varied_param, save_dir, batch_size, n_zbin, fid_dir, pos_subdir, she_subdir, pos_she_subdir, ell_filename, pos_nl_path, she_nl_path, noise_ell_path, lmax, lmin)
```

e) Plot the results using `posteriors.plot_post_max`:

```python
[python]

input_filemask = 'output-directory-from-previous-step/[0-9]*_[0-9]*.txt'
save_path = 'plots-dir/plot-filename.pdf' # Or leave this out to just display plot

plot_post_max(input_filemask, save_path)
```

## Figure 2: Histograms of Wishart and Gaussian 1D posterior mean and standard deviation

a) As Fig. 1 steps (a)-(c).

b) Loop over observations, run the Gaussian and Wishart likelihoods for each and determine the mean and standard deviation, using `run_likelihoods.post_mean_std_1d`:

```python
[python]

obs_filemask = 'path-to-output-from-1c/[0-9]*.npz'
theory_filemask = 'path-to-grid-directory/_[0-9]*/'
varied_param = 'w'
save_dir = 'directory-to-save-results-to'
batch_size = 1000
n_zbin = 5
fid_dir = 'path-to-fiducial-Cls-directory'
pos_subdir = 'galaxy_cl'
she_subdir = 'shear_cl'
pos_she_subdir = 'galaxy_shear_cl'
ell_filename = 'ell.txt'
pos_nl_path = 'path-to-pos_nl'
she_nl_path = 'path-to-she_nl'
noise_ell_path = 'path-to-noise-ell'
lmax = 2000
lmin = 2

post_mean_std_1d(theory_filemask, obs_filemask, varied_param, save_dir, batch_size, n_zbin, fid_dir, pos_subdir, she_subdir, pos_she_subdir, ell_filename, pos_nl_path, she_nl_path, noise_ell_path, lmax, lmin)
```

e) Plot the results using `posteriors.plot_post_mean_std`:

```python
[python]

input_filemask = 'output-directory-from-previous-step/[0-9]*_[0-9]*.txt'
save_path = 'plots-dir/plot-filename.pdf' # Or leave this out to just display plot

plot_post_mean_std(input_filemask, save_path)
```

## Figure 3: 3D posterior triangle plot

a) Produce 3D grid of w, wa, omega_m: as Fig. 1 step (a) except:

1. For the input to `cosmosis_utils.generate_chain_input`, use

    ```python
    params = {
        'cosmological_parameters--w': {
            'min': -1.015,
            'max': -0.985,
            'steps': 51
        },
        'cosmological_parameters--wa': {
            'min': -0.07,
            'max': 0.07,
            'steps': 51
        },
        'cosmological_parameters--omega_m': {
            'min': 0.3131,
            'max': 0.3146,
            'steps': 51
        }
    }
    ```

2. In `tomo_3x2_pipeline.ini`, replace `values = de_grid.ini` with

   ```ini
   values = w0_wa_omm_grid.ini
   ```

b) Produce noise power spectra - as Fig. 1 step (b).

c) Produce mock observed Cls with `simulations.single_obs_cls`:

```python
[python]

n_zbin = 5
fid_dir = 'path-to-fiducial-Cls-directory'
pos_pos_in_dir = fid_dir + '/galaxy_cl'
she_she_in_dir = fid_dir + '/shear_cl'
pos_she_in_dir = fid_dir + '/galaxy_shear_cl'
lmax = 2000
lmin_in = 2
pos_nl_path = 'path-to-pos_nl'
she_nl_path = 'path-to-she_nl'
nside = 1024
lmin_out = 2
mask_path = None # full sky

single_obs_cls(n_zbin, pos_pos_in_dir, she_she_in_dir, pos_she_in_dir, lmax, lmin_in, pos_nl_path, she_nl_path, nside, lmin_out, mask_path)
```

d) Run Wishart and Gaussian likelihoods with `run_likelihoods.run_likes_cl_wishart_gauss`:

```python
[python]

grid_dir = 'path-to-grid-from-step-a'
varied_params = ['w', 'wa', 'omega_m']
save_path = 'path-to-save-likelihood.txt'
n_zbin = 5
obs_dir = 'path-to-output-directory-from-step-c'
obs_pos_pos_dir = obs_dir + '/galaxy_cl'
obs_she_she_dir = obs_dir + '/shear_cl'
obs_pos_she_dir = obs_dir + '/galaxy_shear_cl'
pos_nl_path = 'path-to-pos_nl'
she_nl_path = 'path-to-she_nl'
noise_ell_path = 'path-to-noise-ell'
fid_dir = 'path-to-fiducial-Cls-directory'
fid_pos_pos_dir = fid_dir + '/galaxy_cl'
fid_she_she_dir = fid_dir + '/shear_cl'
fid_pos_she_dir = fid_dir + '/galaxy_shear_cl'
lmax = 2000

run_likelihoods.run_likes_cl_wishart_gauss(grid_dir, varied_params, save_path, n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir, pos_nl_path, she_nl_path, noise_ell_path, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax)
```

e) Produce 3D posterior grids using `posteriors.grid_3d`:

```python
[python]

log_like_path = 'path-to-output-from-step-d'
save_path = 'path-to-save-grid-to.npz'

posteriors.grid_3d(log_like_path, save_path)
```

f) Plot 3D posteriors as a triangle plot using `posteriors.plot_3d`:

```python
[python]

grid_path = 'path-to-output-from-step-e'
contour_levels_sig = [1, 2, 3]
# All the optional parameters can be tweaked as required

posteriors.plot_3d(grid_path, contour_levels_sig)
```

## Figure 4: 2D posterior with discrepant fiducial parameters

a) Produce 2D grid of w, wa - as Fig. 1 step (a) except:

1. For the input to `cosmosis_utils.generate_chain_input`, use

    ```python
    params = {
        'cosmological_parameters--w': {
            'min': -1.016,
            'max': -0.984,
            'steps': 41
        },
        'cosmological_parameters--wa': {
            'min': -0.055,
            'max': 0.055,
            'steps': 41
        }
    }
    ```

b) Produce noise power spectra - as Fig. 1 step (b).

c) Produce full-sky mock observation - as Fig. 3 step (c).

d) Run Wishart and Gaussian likelihoods - as Fig. 3 step (d), except:

1. For the input to `run_likelihoods.run_likes_cl_wishart_gauss`, use

    ```python
    varied_params = ['w', 'wa']
    fid_dir = 'path-to-new-fiducial-directory' # For this plot the fiducial parameters were (w0 = -0.984, wa = 0.055)
    ```

e) Produce 2D posterior plot using `posteriors.plot_2d`:

```python
[python]

like_paths = ['path-to-output-from-step-d']
labels = [['Wishart', 'Gaussian']]
colours = [['C0', 'C1']]
linestyles = [['-', '--']]
contour_levels_sig = [1, 2, 3]
smooth_sigma = 0.8
x_label = r'$w_0$'
y_label = r'$w_a$'
x_lims = (-1.0129, -0.98)
y_lims = (-0.032, 0.077)
true_params = (-1.0, 0.0)
fid_params = (-0.984, 0.055)

posteriors.plot_2d(like_paths, labels, colours, linestyles, contour_levels_sig, smooth_sigma=smooth_sigma, x_label=x_label, y_label=y_label, x_lims=x_lims, y_lims=y_lims, true_params=true_params, fid_params=fid_params)
```

## Figure 5: 2D posterior with lmax = 20

a) As Fig. 4 steps (a)-(e), except:

1. A broader grid is required in the input to `cosmosis_utils.generate_chain_input`:

    ```python
    params = {
        'cosmological_parameters--w': {
            'min': -1.45,
            'max': -0.55,
            'steps': 41
        },
        'cosmological_parameters--wa': {
            'min': -2.8,
            'max': 1.4,
            'steps': 41
        }
    }
    ```

2. For the input to `run_likelihoods.run_likes_cl_wishart_gauss`, use

    ```python
    lmax = 20
    fid_dir = 'path-to-original-fiducial-directory'
    ```

3. In the configuration for `posteriors.plot_2d`:

    ```python
    smooth_sigma = 0.5
    true_params = None
    fid_params = None
    x_lims = (-1.4, -0.6)
    y_lims = (-1.7, 1.5)
    annot = r'$\ell_\mathrm{max} = 20$'
    ```

## Figure 6: 2D posterior with x100 noise

a) As Fig. 4 steps (a)-(e), except:

1. A broader grid (but less broad than Fig. 5) is required in the input to `cosmosis_utils.generate_chain_input`:

    ```python
    params = {
        'cosmological_parameters--w': {
            'min': -1.15,
            'max': -0.81,
            'steps': 41
        },
        'cosmological_parameters--wa': {
            'min': -0.99,
            'max': 0.61,
            'steps': 41
        }
    }
    ```

2. For the input to `simulation.noise_cls`, use

    ```python
    gals_per_sq_arcmin = 30 / n_zbin / 100
    ```

2. For the input to `run_likelihoods.run_likes_cl_wishart_gauss`, use

    ```python
    pos_nl_path = 'path-to-output-from-step-a2'
    she_nl_path = 'path-to-output-from-step-a2'
    fid_dir = 'path-to-original-fiducial-directory'
    ```

3. In the configuration for `posteriors.plot_2d`:

    ```python
    smooth_sigma = 0.5
    true_params = None
    fid_params = None
    x_lims = (-1.18, -0.84)
    y_lims = (-0.6, 0.6)
    annot = r'$\times 100$ noise'
    ```

## Figure 7: Illustration of l -> l_eff mapping

a) Produce theory Cls for a single cosmology - as Fig. 1 step (a), except:

1. Use the `grid` sampler in `tomo_3x2_pipeline.ini`:

    ```ini
    [runtime]
    sampler = grid

    [grid]
    nsample_dimension = 1
    save = /path-to-output-dir/

2. Set `de_grid.ini` to use fixed values of `w` and `wa`:

    ```ini
    w = -1.0
    wa = 0.0
    ```

Note: for ad-hoc single-cosmology runs like this, it can be easier to use [CCL](https://github.com/LSSTDESC/CCL) than CosmoSIS.

b) Produce harmonic space window functions for each l, using [pseudo_cl_likelihood](https://github.com/robinupham/pseudo_cl_likelihood)`.mask_to_w.mask_to_w`:

```python
[python]

mask_path = 'path-to-full-euclid-like-mask.fits.gz'
lmax = 200
save_dir = 'path-to-output-directory'

pseudo_cl_likelihood.mask_to_w.mask_to_w(mask_path, lmax, do_wpm=False)
```

c) Combine harmonic space window functions into a single file, using [pseudo_cl_likelihood](https://github.com/robinupham/pseudo_cl_likelihood)`.mask_to_w.combine_w_files`:

```python
[python]

filemask = 'path-to-output-directory-from-step-b/w0_{l}.npz'
save_path = 'path-to-save-combined-file.npz'
l_start = 2

pseudo_cl_likelihood.mask_to_w.combine_w_files(filemask, save_path, l_start)
```

d) Calculate marginal pseudo-alm covariance for each l, using `pcl_like.marginal_alm_cov`:

```python
[python]

cl_path = 'path-to-theory-power-spectrum-from-step-a.txt'
w0_path = 'path-to-w0-from-step-c.npz'
lmax_in = 200
lmin_in = 2
lmax_out = 100
lmin_out = 2
save_path = 'path-to-save-covariances.npz'

pcl_like.marginal_alm_cov(cl_path, w0_path, lmax_in, lmin_in, lmax_out, lmin_out, save_path)
```

e) Calculate marginal pseudo-Cl likelihood distribution for each l, using `pcl_like.marginal_cl_likes`:

```python
[python]

covs_path = 'path-to-output-from-step-d.npz'
save_path = 'path-to-save-likelihoods.npz'

pcl_like.marginal_cl_likes(covs_path, save_path)
```

f) Calculate and plot skewness with l -> l_eff mapping, using `moments.explain_mapping`:

```python
[python]

cut_sky_pdf_path = 'path-to-output-from-step-e.npz'
lmax = 80

moments.explain_mapping(cut_sky_pdf_path, lmax)
```

## Figure 8: Exact and extrapolated l_eff vs l

a) As Fig. 7 steps (a)-(e).

b) Calculate full-sky and cut-sky skewness and excess kurtosis per l, using `moments.get_all_moments`:

```python
[python]

cut_sky_pdf_path = 'path-to-output-from-fig-7-step-e.npz'
lmax = 80
data_save_path = 'path-to-save-moments.txt'

moments.get_all_moments(cut_sky_pdf_path, lmax, data_save_path=data_save_path)
```

c) Plot exact and extrapolated l -> l_eff mapping with `moments.plot_leff_map`.

```python
[python]

data_path = 'path-to-output-from-step-b.txt'
fit_lmax = 80
lmax = 2000

moments.plot_leff_map(data_path, fit_lmax, lmax)
```

## Figure 9: Skewness and excess kurtosis from simulations compared to l -> leff mapping prediction

a) As Fig. 8 steps (a)-(b).

b) Evaluate l -> l_eff mapping for both skewness and kurtosis using `moments.get_leff_map`:

```python
[python]

data_path = 'path-to-output-from-fig-8-step-b.txt'
fit_lmax = 80
lmax = 2000
map_save_path_kurt = 'path-to-save-kurtosis-mapping.txt'
map_save_path_skew = 'path-to-save-skewness-mapping.txt'

moments.get_leff_map(data_path, fit_lmax, lmax map_save_path_kurt=map_save_path_kurt, map_save_path_skew=map_save_path_skew)
```

c) Produce noise power spectra - as Fig. 1 step (b).

d) Produce single-field simulations using `simulation.sim_singlespec`:

```python
[python]

input_cl_path = 'path-to-theory-power-spectrum-from-fig-7-step-a.txt'
lmax = 2000
lmin_in = 2
nl_path = 'path-to-noise-power-spectrum-from-step-c.txt'
mask_path = 'path-to-mask.fits.gz'
nside = 1024
n_loop = int(1e5)
batch_size = 100
lmin_out = 2
save_dir = 'path-to-output_directory'

simulation.sim_singlespec(input_cl_path, lmax, lmin_in, nl_path, mask_path, nside, n_loop, batch_size, lmin_out, save_dir)
```

e) Calculate skewness and excess kurtosis from simulations using `moments.save_sim_moments`:

```python
[python]

input_filemask = 'path-to-output-directory-from-step-d/batch_*.npz'
batch_size = 100
lmax = 2000
lmin = 2
save_path = 'path-to-save-moments.txt'
whiten = False

moments.save_sim_moments(input_filemask, batch_size, lmax, lmin, save_path, whiten)
```

f) Produce plot comparing simulated with predicted moments using `moments.plot_sim_moments`:

```python
[python]

sim_path = 'path-to-moments-output-by-step-e.txt'
kurt_leff_map_path = 'path-to-kurtosis-leff-map-output-by-step-b.txt'
skew_leff_map_path = 'path-to-skewness-leff-map-output-by-step-b.txt'

moments.plot_sim_moments(sim_path, kurt_leff_map_path, skew_leff_map_path)
```

## Figure 10: 2D posterior using l -> l_eff mapping

a) Obtain kurtosis l -> l_eff mapping - as Fig. 9 steps (a)-(b).

b) Produce 2D theory grid of `w`, `wa` -  as Fig. 4 step (a).

c) Produce noise power spectra - as Fig. 1 step (b).

d) Produce mock observation using l -> l_eff mapping, using `simulation.leff_obs`:

```python
[python]

n_zbin = 5
fid_cl_dir = 'path-to-fiducial-cl-directory' # Can be a subdirectory from the grid produced in step (b)
pos_pos_dir = fid_cl_dir + '/galaxy_cl'
she_she_dir = fid_cl_dir + '/shear_cl'
pos_she_dir = fid_cl_dir + '/galaxy_shear_cl'
ell_filename = 'ell.txt'
pos_nl_path = 'path-to-pos_nl-from-step-c.txt'
she_nl_path = 'path-to-she_nl-from-step-c.txt'
noise_ell_path = 'path-to-noise-ell-from-step-c.txt'
leff_path = 'path-to-kurtosis-leff-map-from-step-a.txt'
save_dir = 'path-to-output-directory'

simulation.leff_obs(n_zbin, pos_pos_dir, she_she_dir, pos_she_dir, ell_filename, pos_nl_path, she_nl_path, noise_ell_path, leff_path, save_dir)
```

d) Evaluate Wishart likelihood using l_eff mapping, using `run_likelihoods.run_like_cl_wishart`:

```python
[python]

grid_dir = 'path-to-grid-output-by-step-b'
varied_params = ['w', 'wa']
save_path = 'path-to-save-likelihood.txt'
n_bin = 5
obs_dir = 'path-to-obs-directory-from-step-c'
obs_pos_pos_dir = obs_dir + '/galaxy_cl'
obs_she_she_dir = obs_dir + '/shear_cl'
obs_pos_she_dir = obs_dir + '/galaxy_shear_cl'
pos_nl_path = 'path-to-pos_nl-from-step-c.txt'
she_nl_path = 'path-to-she_nl-from-step-c.txt'
noise_ell_path = 'path-to-noise-ell-from-step-c.txt'
lmax = 2000
leff_path = 'path-to-kurtosis-leff-map-from-step-a.txt'

run_likelihoods.run_like_cl_wishart(grid_dir, varied_params, save_path, n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir, pos_nl_path, she_nl_path, noise_ell_path, lmax, leff_path=leff_path)
```

e) Evaluate Gaussian likelihood using l_eff mapping, using `run_likelihoods.run_like_cl_gauss`:

```python
[python]

grid_dir = 'path-to-grid-output-by-step-b'
varied_params = ['w', 'wa']
save_path = 'path-to-save-likelihood.txt'
n_bin = 5
obs_dir = 'path-to-obs-directory-from-step-c'
obs_pos_pos_dir = obs_dir + '/galaxy_cl'
obs_she_she_dir = obs_dir + '/shear_cl'
obs_pos_she_dir = obs_dir + '/galaxy_shear_cl'
pos_nl_path = 'path-to-pos_nl-from-step-c.txt'
she_nl_path = 'path-to-she_nl-from-step-c.txt'
noise_ell_path = 'path-to-noise-ell-from-step-c.txt'
fid_cl_dir = 'path-to-fiducial-cl-directory'
fid_pos_pos_dir = fid_cl_dir + '/galaxy_cl'
fid_she_she_dir = fid_cl_dir + '/shear_cl'
fid_pos_she_dir = fid_cl_dir + '/galaxy_shear_cl'
lmax = 2000
leff_path = 'path-to-kurtosis-leff-map-from-step-a.txt'

run_likelihoods.run_like_cl_gauss(grid_dir, varied_params, save_path, n_zbin, obs_pos_pos_dir, obs_she_she_dir, obs_pos_she_dir, pos_nl_path, she_nl_path, noise_ell_path, fid_pos_pos_dir, fid_she_she_dir, fid_pos_she_dir, lmax, leff_path=leff_path)
```

f) Produce 2D posterior plot using `posteriors.plot_2d`:

```python
[python]

like_paths = ['path-to-wishart-like-from-step-d.txt', 'path-to-gaussian-like-from-step-e.txt']
labels = [['Wishart'], ['Gaussian']]
colours = [['C0'], ['C1']]
linestyles = [['-'], ['--']]
contour_levels_sig = [1, 2, 3]
smooth_sigma = 0.7
x_label = r'$w_0$'
y_label = r'$w_a$'
x_lims = (-1.013, -0.982)
y_lims = (-0.05, 0.06)
annot = 'Mock cut-sky setup'

posteriors.plot_2d(like_paths, labels, colours, linestyles, contour_levels_sig, smooth_sigma=smooth_sigma, x_label=x_label, y_label=y_label, x_lims=x_lims, y_lims=y_lims, annot=annot)
```

## Figure 11: Mutual information for all pairs, full-sky vs cut-sky

a) Produce theory Cls for a single cosmology - as Fig. 7 step (a).

b) Produce noise power spectra - as Fig. 1 step (b).

c) Generate full-sky and cut-sky simulated bandpowers using `simulation.sim_bps`:

```python
[python]

n_zbin = 5
input_cl_dir = 'path-to-theory-cls-from-step-a'
pos_pos_in_dir = input_cl_dir + '/galaxy_cl'
she_she_in_dir = input_cl_dir + '/shear_cl'
pos_she_in_dir = input_cl_dir + '/galaxy_shear_cl'
lmax = 2000
lmin_in = 2
pos_nl_path = 'path-to-pos_nl-from-step-b.txt'
she_nl_path = 'path-to-she_nl-from-step-b.txt'
nside = 1024
n_bandpower = 10
lmin_out = 2
mask_path = 'path-to-full-Euclid-like-mask.fits.gz'
n_loop = 50000
batch_size = 1000
save_dir = 'path-to-output-dir'

simulation.sim_bps(n_zbin, pos_pos_in_dir, she_she_in_dir, pos_she_in_dir, lmax, lmin_in, pos_nl_path, she_nl_path, nside, n_bandpower, lmin_out, mask_path, n_loop, batch_size, save_dir)
```

d) Combine simulated bandpower batches into one file each for full-sky and cut-sky, using `simulation.combine_sim_bp_batches`:

```python
[python]

# Full sky
input_mask = 'path-to-output-dir-from-step-c/fs_{batch}.npz'
simulation.combine_sim_bp_batches(input_mask)

# Cut sky
input_mask = 'path-to-output-dir-from-step-c/ma_{batch}.npz'
simulation.combine_sim_bp_batches(input_mask)
```

Note: [The public data on Zenodo](https://doi.org/10.1093/mnras/stab522) is simply a combination of the full-sky and cut-sky output from this step.

e) Calculate pairwise whitened mutual information for all pairs, using `mutual_info.calculate_all_mi`:

```python
[python]

data_path_fs = 'path-to-full-sky-output-from-step-d.npz'
data_path_ma = 'path-to-cut-sky-output-from-step-d.npz'
n_fields = 10
n_bandpowers = 10
save_path = 'path-to-save-output_{pop}.npz'

mutual_info.calculate_all_mi(data_path_fs, data_path_ma, n_fields, n_bandpowers, save_path)
```

f) Combine single-pair-population output into a single file, using `mutual_info.combine_mi_files`:

```python
[python]

input_filemask = 'path-to-output-from-step-e_*.npz'
save_path = 'path-to-save-combined-file.npz'

mutual_info.combine_mi_files(input_filemask, save_path)
```

g) Plot mutual information distribution histograms, using `mutual_info.plot_all_pairs`:

```python
[python]

input_path = 'path-to-combined-file-output-by-step-f.npz'

mutual_info.plot_all_pairs(input_path)
```

## Figure 12: Mutual information compared to Gaussian samples

a) Produce mutual information estimates from simulations - as Fig. 11 steps (a)-(f).

b) Generate matching Gaussian samples, using `simulation.gaussian_samples`:

```python
[python]

# Full sky
sim_bp_path = 'path-to-full-sky-sim-bp-from-fig-11-step-d.npz'
save_path = 'path-to-save-full-sky-gaussian-samples.npz'
simulation.gaussian_samples(sim_bp_path, save_path)

# Cut sky
sim_bp_path = 'path-to-cut-sky-sim-bp-from-fig-11-step-d.npz'
save_path = 'path-to-save-cut-sky-gaussian-samples.npz'
simulation.gaussian_samples(sim_bp_path, save_path)
```

c) Calculate pairwise whitened mutual information of Gaussian samples for all pairs, using `mutual_info.calculate_all_mi`:

```python
[python]

data_path_fs = 'path-to-full-sky-gaussian-samples-from-step-b.npz'
data_path_ma = 'path-to-cut-sky-gaussian-samples-from-step-b.npz'
n_fields = 10
n_bandpowers = 10
save_path = 'path-to-save-output_{pop}.npz'
data_label = 'gauss_bp'

mutual_info.calculate_all_mi(data_path_fs, data_path_ma, n_fields, n_bandpowers, save_path, data_label=data_label)
```

d) Combine single-pair-population output into a single file, using `mutual_info.combine_mi_files`:

```python
[python]

input_filemask = 'path-to-output-from-step-c_*.npz'
save_path = 'path-to-save-combined-file.npz'

mutual_info.combine_mi_files(input_filemask, save_path)
```

e) Plot mutual information distribution histograms, using `mutual_info.plot_sim_gauss`:

```python
[python]

sim_path = 'path-to-simulated-mi-from-fig-11-step-f.npz'
gauss_path = 'path-to-gaussian-sample-mi-from-step-d.npz'

mutual_info.plot_sim_gauss(sim_path, gauss_path)
```

## Figure 13: Mutual information distributions for parent-child pairs

a) Produce mutual information estimates from simulations - as Fig. 11 steps (a)-(f).

b) Plot histograms of mutual information distributions for parent-child pairs, using `mutual_info.plot_parents`:

```python
[python]

input_path = 'path-to-mi-estimates-from-fig-11-step-f.npz'

mutual_info.plot_parents(input_path)
```

## Figure 14: Mutual information distributions for adjacent bandpowers, with and without whitening

a) Produce whitened mutual information estimates from simulations - as Fig. 11 steps (a)-(f).

b) Produce unwhitened mutual information estimates from simulations - repeat Fig. 11 steps (e)-(f), except:

1. In the input to `mutual_info.calculate_all_mi`, set

    ```python
    no_whiten = True
    ```

c) Plot histograms of mutual information with and without whitening, using `mutual_info.plot_whitening`:

```python
[python]

nowhiten_path = 'path-to-unwhitened-mi-from-step-b.npz'
whiten_path = 'path-to-whitened-mi-from-step-a.npz'
n_fields = 10
n_bandpowers = 10

mutual_info.plot_whitening(nowhiten_path, whiten_path, n_fields, n_bandpowers)
```

## Figure 15: Mutual information vs l for same-bandpower pairs

a) Produce mutual information estimates from simulations - as Fig. 11 steps (a)-(f).

b) Generate bandpower binning matrix, using `simulation.get_binning_matrix`:

```python
[python]

n_bandpowers = 10
output_lmin = 2
output_lmax = 2000
pbl_save_path = 'path-to-save-pbl.txt'
pbl_header = f'Bandpower binning matrix for {n_bandpowers} log-spaced bandpowers from {output_lmin} to {output_lmax}'

pbl = simulation.get_binning_matrix(n_bandpowers, output_lmin, output_lmax)

np.savetxt(pbl, pbl_save_path, header=pbl_header)
```

c) Plot mutual information vs l for same-bandpower pairs, using `mutual_info.plot_vs_l`:

```python
[python]

input_path = 'path-to-mi-estimates-from-step-a.npz'
pbl_path = 'path-to-pbl-from-step-b.txt'
lmin = 2

mutual_info.plot_vs_l(input_path, pbl_path, lmin)
```

## Figure 16: Transcovariance distributions

a) Generate full-sky and cut-sky simulated bandpowers - as Fig. 11 steps (a)-(d).

b) Calculate transcovariance using `mutual_info.sim_bp_transcov`:

```python
[python]

data_path_fs = 'path-to-full-sky-bandpowers-from-step-a.npz'
data_path_ma = 'path-to-cut-sky-bandpowers-from-step-a.npz'
n_fields = 10
n_bandpowers = 10
save_path = 'path-to-save-transcovariance.npz'
pit = False # or True to isolate non-Gaussian dependence

mutual_info.sim_bp_transcov(data_path_fs, data_path_ma, n_fields, n_bandpowers, save_path, pit=pit)
```

c) Generate bandpower binning matrix - as Fig. 15 step (b).

d) Plot transcovariance distributions using `mutual_info.plot_transcov`:

```python
[python]

input_path = 'path-to-transcovariance-from-step-b.npz'

mutual_info.plot_transcov(input_path)
```

## Figure 17: Skewness of SLICS vs Gaussian fields

a) Combine all SLICS Cl files for each tomographic bin into a single file, using `simulation.combine_slics`:

```python
[python]

input_filemask = 'path-to-SLICS/2pt-functions/KiDS450/l2Cl/l2cl_kappa_nz_KiDS450_tomo{tomo}.dat_LOS{los}'
save_filemask = 'output-path_tomo{tomo}.npz'
lmax = 5000

for tomo in range(5):
    simulation.combine_slics(input_filemask, tomo, save_filemask, lmax)
```

b) Produce theory Cls for a single cosmology - as Fig. 7 step (a).

c) Produce flat-sky Gaussian field simulations, using `simulation.sim_flat`:
```python
[python]

n_zbin = 5
cl_filemask = 'path-to-theory-cls/bin_{j}_{i}.txt'
input_lmin = 2
nx = 700
ny = 700
lx = 10 * np.pi/180
ly = 10 * np.pi/180
leff_min = 35
leff_max = 4970
bin_size = 35
n_real = 948
save_path = 'path-to-save-sim-cls.npz'

simulation.sim_flat(n_zbin, cl_filemask, input_lmin, nx, ny, lx, ly, leff_min, leff_max, bin_size, n_real, save_path)
```

d) Plot skewness against l, using `moments.plot_skew_slics_gauss`:

```python
[python]

slics_filemask = 'path-to-combined-slics-from-step-a_tomo{tomo}.npz'
gauss_path = 'path-to-gaussian-field-cls-from-step-c.npz'
n_zbin = 5

moments.plot_skew_slics_gauss(slics_filemask, gauss_path, n_zbin)
```

## Figure 18: Mutual information of SLICS vs Gaussian fields

a) Prepare SLICS and Gaussian field simulated bandpowers - as Fig. 17 steps (a)-(c).

b) Estimate mutual information for SLICS and Gaussian field simulations using `mutual_info.calculate_mi_slics_gauss`:

```python
[python]

slics_filemask = 'path-to-combined-slics-from-fig-17-step-a_tomo{tomo}.npz'
gauss_path = 'path-to-gaussian-field-cls-from-fig-17-step-c.npz'
n_tomo = 5
n_bp = 142
n_real = 948
save_path = 'path-to-save-mi_{pop}.npz'

mutual_info.calculate_mi_slics_gauss(slics_filemask, gauss_path, n_tomo, n_bp, n_real, save_path)
```

c) Combine single-pair-population output into a single file, using `mutual_info.combine_mi_slics_gauss`:

```python
[python]

input_filemask = 'path-to-output-from-step-b_*.npz'
save_path = 'path-to-save-combined-file.npz'

mutual_info.combine_mi_slics_gauss(input_filemask, save_path)
```

d) Plot mutual information distributions for all pairs and same-spectrum pairs, using `mutual_info.plot_samespec_slics_gauss`:

```python
[python]

input_path = 'path-to-mi-estimates-from-step-c.npz'

mutual_info.plot_samespec_slics_gauss(input_path)
```

## Figure 19: Mutual information matrix for SLICS vs Gaussian fields

a) Calculate mutual information estimates for SLICS and Gaussian fields - as Fig. 18 steps (a)-(c).

b) Plot mutual information matrix, using `mutual_info.plot_matrix_slics_gauss`:

```python
[python]

input_path = 'path-to-mi-estimates-from-fig-18-step-c.npz'

mutual_info.plot_matrix_slics_gauss(input_path)
```
