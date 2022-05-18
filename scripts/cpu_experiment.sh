#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#!
#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J cpu_experiment
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A KRUEGER-SL2-CPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! How much wallclock time will be required?
#SBATCH --time=24:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#SBATCH --output=slurm-out/%x.%j.out

#/bin/bash
set -x # echo on

for i in {1..1}; do python3 1d_regression.py --tag=cpu_experiment --early_stopping=False --num_epochs=50000 --lr_schedule=cosine --optimiser=sgd --nonlinearity=relu --generalisation_task=interpolation --normalise=True --adjust_data_linearly=True --dataset=sine --num_datapoints=100 --model=AsiShallowRelu --hidden_units=1000 --learning_rate=0.001; done
