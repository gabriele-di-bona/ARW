#!/bin/bash
#$ -cwd
#$ -t 2002-2010
#$ -j y
#$ -pe smp 1
#$ -l h_vmem=10G
#$ -l highmem
#$ -l h_rt=240:0:0
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1

conda activate ARW
cd ../../../codes # launching from subdirectory ./outputs
# cd ../../codes # launching from bash_script specific directory
python fig_new_SM_multicores_random_start_random_alpha.py -ID $(( ${SGE_TASK_ID} - 1 )) -c ${NSLOTS} -a 0 -M 1000
