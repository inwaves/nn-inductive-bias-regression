#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#!
#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J sine-optimiser-ablation
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A KRUEGER-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 3 cpus per GPU.
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=05:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#SBATCH --output=slurm-out/%x.%j.out


#! Do not change:
#SBATCH -p ampere
#/bin/bash
set -x #echo on
start=$(date +%s)

python3 1d_regression.py --dataset=sine --tag=sine-optimiser-ablation --optimiser=sgd --nonlinearity=relu --generalisation_task=baseline --model=ASIShallowRelu --hidden_units=10 --learning_rate=0.1 --adjust_data_linearly=True
python3 1d_regression.py --dataset=sine --tag=sine-optimiser-ablation --optimiser=sgd --nonlinearity=relu --generalisation_task=baseline --model=ASIShallowRelu --hidden_units=100 --learning_rate=0.01 --adjust_data_linearly=True
python3 1d_regression.py --dataset=sine --tag=sine-optimiser-ablation --optimiser=sgd --nonlinearity=relu --generalisation_task=baseline --model=ASIShallowRelu --hidden_units=1000 --learning_rate=0.001 --adjust_data_linearly=True

python3 1d_regression.py --dataset=sine --tag=sine-optimiser-ablation --optimiser=adam --nonlinearity=relu --generalisation_task=baseline --model=ASIShallowRelu --hidden_units=10 --learning_rate=0.1 --adjust_data_linearly=True
python3 1d_regression.py --dataset=sine --tag=sine-optimiser-ablation --optimiser=adam --nonlinearity=relu --generalisation_task=baseline --model=ASIShallowRelu --hidden_units=100 --learning_rate=0.01 --adjust_data_linearly=True
python3 1d_regression.py --dataset=sine --tag=sine-optimiser-ablation --optimiser=adam --nonlinearity=relu --generalisation_task=baseline --model=ASIShallowRelu --hidden_units=1000 --learning_rate=0.001 --adjust_data_linearly=True

python3 1d_regression.py --dataset=sine --tag=sine-optimiser-ablation --optimiser=momentum --nonlinearity=relu --generalisation_task=baseline --model=ASIShallowRelu --hidden_units=10 --learning_rate=0.1 --adjust_data_linearly=True
python3 1d_regression.py --dataset=sine --tag=sine-optimiser-ablation --optimiser=momentum --nonlinearity=relu --generalisation_task=baseline --model=ASIShallowRelu --hidden_units=100 --learning_rate=0.01 --adjust_data_linearly=True
python3 1d_regression.py --dataset=sine --tag=sine-optimiser-ablation --optimiser=momentum --nonlinearity=relu --generalisation_task=baseline --model=ASIShallowRelu --hidden_units=1000 --learning_rate=0.001 --adjust_data_linearly=True

end=$(date +%s)

runtime=$((end-start))