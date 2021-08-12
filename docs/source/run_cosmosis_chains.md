run\_cosmosis\_chains.sh
========================

Bash script to run CosmoSIS chains in parallel using the list sampler.

Designed to be used alongside the `tomo_3x2_pipeline.ini` pipeline, which supports the `$CHAIN_NO` environment variable, and to use chains produced by `cosmosis_utils.generate_chain_input`.

To use:

1. Produce chain input using `cosmosis_utils.generate_chain_input`.

2. Set up pipeline by setting input and output directories and any other required settings in `tomo_3x2_pipeline.ini`.

3. Set the paths to CosmoSIS and to the pipeline file in `run_cosmosis_chains.sh`.

4. Run:

    ```bash
    bash run_cosmosis_chains.sh
    ```
