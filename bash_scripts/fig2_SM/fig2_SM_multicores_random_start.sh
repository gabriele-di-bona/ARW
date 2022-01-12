#!/bin/bash
#$ -cwd
#$ -t 1-6000 # Unfortunately this has to start from 1.
#$ -j y
#$ -pe smp 1
#$ -l h_vmem=10G
# #$ -l highmem
#$ -l h_rt=240:0:0
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1

conda activate ARW
cd ../../codes
python fig2_SM_multicores_random_start.py -ID $(( ${SGE_TASK_ID} - 1 )) -c 1
