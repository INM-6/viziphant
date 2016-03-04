#!/bin/bash
#SBATCH --job-name corrcoeff_spinest
#SBATCH --array 0-1
# Redirect stdout and stderr:
#SBATCH --output=../../qsub/out/corrcoeff_spinest_%j.out
# Redirect stderr:
#SBATCH --error=../../qsub/err/corrcoeff_spinest_%j.err
#SBATCH --workdir ./
# Send mail notifications
#SBATCH --mail-type=END
# load python module
module load pystuff
# activate my env
source activate venv
# run my python script
srun python corrcoeff_cluster_spinnest.py
