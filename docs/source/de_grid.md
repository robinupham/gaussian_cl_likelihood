de\_grid.ini
============

CosmoSIS values file to produce a 2D grid of `w` and `wa`, with all other parameters fixed to Planck 2018 best-fit values.

Designed for use with the `list` sampler, in which case the specified ranges of `w` and `wa` act as bounds on the allowed values of these parameters. The supplied default ranges are deliberately extremely broad for this reason. Can also be used with the `grid` sampler, but in that case the ranges of `w` and `wa` should be explicitly set to the required ranges.
