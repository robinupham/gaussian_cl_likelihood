structure/projection/project\_2d.py
===================================

Modified version of the `project_2d` CosmoSIS standard library module, allowing a `linspaced` boolean option to produce
linearly spaced ells (default `False`, i.e. log-spaced ells) in lines 374-378.

To use this version, replace the usual `cosmosis-standard-library/structure/projection/project_2d.py` file with this one,
then in the `[project_2d]` section of your CosmoSIS pipeline ini file, add the line:

```ini
linspaced = T
```

See the original `project_2d` documentation at [https://bitbucket.org/joezuntz/cosmosis/wiki/default_modules/project_2d_1.0](https://bitbucket.org/joezuntz/cosmosis/wiki/default_modules/project_2d_1.0).
