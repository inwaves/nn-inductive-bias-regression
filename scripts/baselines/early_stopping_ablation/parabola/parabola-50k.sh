#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#!
#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J pa-no-es-50000
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A [REDACTED]
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! How much wallclock time will be required?
#SBATCH --time=36:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#SBATCH --output=slurm-out/%x.%j.out


#! Do not change:
#SBATCH -p icelake-himem
#/bin/bash
set -x #echo on
start=$(date +%s)
num_iter=1

for ((i=1;i<=num_iter;i++))
do
  python3 1d_regression.py --tag=pa-no-es-50000 --early_stopping=False --num_epochs=100000 --lr_schedule=plateau --dataset=parabola --generalisation_task=baseline --model_type=ASIShallowRelu --hidden_units=50000 --learning_rate=0.0001 --adjust_data_linearly=True
done

end=$(date +%s)

runtime=$((end-start))

echo $runtime
