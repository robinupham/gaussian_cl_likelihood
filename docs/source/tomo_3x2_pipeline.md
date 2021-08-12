tomo\_3x2\_pipeline.ini
=======================

CosmoSIS pipeline to produce tomographic 3x2pt Cls.

See the CosmoSIS documentation at [https://bitbucket.org/joezuntz/cosmosis/wiki/Home](https://bitbucket.org/joezuntz/cosmosis/wiki/Home).

This pipeline is designed to use the `list` sampler, with input as chains produced by `cosmosis_utils.generate_chain_input`. It is designed to be run using `run_cosmosis_chains.sh`.

To use:

1. Produce chain input using `cosmosis_utils.generate_chain_input`.

2. Set up pipeline by setting input and output directories and any other required settings in `tomo_3x2_pipeline.ini`.

3. Set the paths to CosmoSIS and to the pipeline file in `run_cosmosis_chains.sh`.

4. Run `run_cosmosis_chains.sh`:

    ```bash
    bash run_cosmosis_chains.sh
    ```
