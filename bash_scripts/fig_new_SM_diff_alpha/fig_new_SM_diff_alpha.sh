#!/bin/bash
#$ -cwd
#$ -t 1001-2500 # do URW and MERW (up to 10800) only for alpha=0.1
#$ -j y
#$ -pe smp 1
#$ -l h_vmem=2G
# #$ -l highmem
#$ -l h_rt=240:0:0
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1 #${NSLOTS}

conda activate ARW
cd ../../../codes # launching from subdirectory ./outputs
# cd ../../codes # launching from bash_script specific directory
python fig_new_SM_multicores_random_start_diff_alpha.py -ID $(( ${SGE_TASK_ID} - 1 )) -c ${NSLOTS} -a $1
