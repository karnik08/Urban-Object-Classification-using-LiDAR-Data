#!/bin/bash
#
#SBATCH --job-name=voxnet_preprocess
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
##SBATCH --mail-user=kp2670@nyu.edu 
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --output=./voxnet_preprocess.out
#SBATCH --error=./voxnet_preprocess.err

singularity exec --overlay $SCRATCH/singular/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c " source /ext3/env.sh;python3 voxnet_preprocess.py" 
