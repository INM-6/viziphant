#!/bin/bash
#SBATCH --job-name cch_spinest
#SBATCH --array 0-100
#SBATCH --workdir=$HOME/projects/
# Redirect stdout and stderr:
#SBATCH --output=$HOME/qsub/out/cch_spinest%j.err
# Redirect stderr:
#SBATCH --error=$HOME/qsub/err/cch_spinest %j.err
# Send mail notifications
#SBATCH --mail-type=END
# load python module
module load pystuff
# activate my env
source activate venv
# run my python script
srun python cch_cluster_spinnaker.py
