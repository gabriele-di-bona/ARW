#!/bin/bash
#$ -cwd
#$ -t 1-1797 # Unfortunately this has to start from 1.
#$ -j y
#$ -pe smp 1
#$ -l h_vmem=20G
# #$ -l highmem
#$ -l h_rt=240:0:0
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1

conda activate ARW
cd ../../codes
python fig2_diff_N_multicores_random_start.py -ID $(( ${SGE_TASK_ID} - 1 )) -c 1
