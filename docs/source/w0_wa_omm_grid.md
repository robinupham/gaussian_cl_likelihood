w0\_wa\_omm_grid.ini
====================

CosmoSIS values file to produce a 3D grid of `w`, `wa` and `omega_m`, with all other parameters fixed to Planck 2018 best-fit values.

Designed for use with the `list` sampler, in which case the specified parameter ranges act as bounds on the allowed values of these parameters. The supplied default ranges are deliberately extremely broad for this reason. Can also be used with the `grid` sampler, but in that case the parameter ranges should be explicitly set to the required ranges.
