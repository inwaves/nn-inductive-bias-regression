#/bin/bash
#SBATCH --account KRUEGER-SL3-GPU
#SBATCH --pascal pascal
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

python3 1d_regression.py --dataset=sine --generalisation_task=baseline --model=ASIShallowRelu --hidden_units=10 --learning_rate=0.001