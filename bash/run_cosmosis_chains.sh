#!/bin/bash

# Bash script to submit N_CHAINS cosmosis chains using nohup,
# with output from each chain to a separate log file.
# Pipeline must be set up to use the CHAIN_NO environment variable.

# Parameters to set before using
N_CHAINS=11
COSMOSIS_DIR=/path-to-cosmosis
PIPELINE_PATH=/path-to-pipeline.ini

# Setup
cd $COSMOSIS_DIR
source config/setup-cosmosis

# Loop
for ((CHAIN_NO = 0; CHAIN_NO < $N_CHAINS; CHAIN_NO++))
do
   CHAIN_LOG="chain$CHAIN_NO.out"
   export CHAIN_NO
   nohup cosmosis $PIPELINE_PATH > $CHAIN_LOG 2>&1 &
   echo "Chain $CHAIN_NO running with PID $!"
done

# Tidy up
unset CHAIN_NO
echo "Done"
