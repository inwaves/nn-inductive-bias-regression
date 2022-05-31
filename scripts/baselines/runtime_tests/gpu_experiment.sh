set -x # echo on

for i in {1..1}; do python3 1d_regression.py --early_stopping=False --num_epochs=20000 --a_w=1 --a_b=5 --optimiser=sgd --nonlinearity=relu --generalisation_task=baseline --normalise=True --adjust_data_linearly=True --dataset=sine --num_datapoints=10 --model_type=ASIShallowRelu --hidden_units=100 --learning_rate=0.01 --init="uniform"; done
