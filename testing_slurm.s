#!/bin/bash
#
#SBATCH --job-name=ml_check
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=END
##SBATCH --mail-user=kp2670@nyu.edu 
#SBATCH --time=100:00:00
#SBATCH --mem=32GB
#SBATCH --output=./ml_check.out
#SBATCH --error=./ml_check.err
singularity exec --overlay $SCRATCH/singular/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh;python3 kp_lidar_data_processing_ml_normalized.py" 
