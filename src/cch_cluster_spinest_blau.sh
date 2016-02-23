#!/bin/bash
#SBATCH --output=../qsub/out/findSpikePatterns%j.out
# Redirect stderr:
#SBATCH --error=../qsub/err/findSpikePatterns%j.err
# Send mail notifications
#SBATCH --mail-type=ALL
# load python module
module load pystuff
# activate my env
source activate venv
# run my python script
python -B cch_cluster_spinnaker.py

