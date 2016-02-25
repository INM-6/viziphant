#!/bin/bash
#SBATCH --job-name cch_spinest
#SBATCH --array 0-99
# Redirect stdout and stderr:
#SBATCH --output=../../qsub/out/cch_spinest_%j.out
# Redirect stderr:
#SBATCH --error=../../qsub/err/cch_spinest_%j.err
#SBATCH --workdir ./
# Send mail notifications
#SBATCH --mail-type=END
# load python module
module load pystuff
# activate my env
source activate venv
# run my python script
python cch_cluster_spinnest.py
