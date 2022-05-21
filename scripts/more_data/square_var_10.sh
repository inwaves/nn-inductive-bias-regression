set -x #echo on
start=$(date +%s)
num_iter=1

for ((i=1;i<=num_iter;i++))
do
  python3 1d_regression.py --tag=relu-sq10 --dataset=square --generalisation_task=baseline --model_type=ShallowRelu --hidden_units=10 --learning_rate=0.1 --adjust_data_linearly=True --early_stopping=True --num_epochs=100000 --num_datapoints=50
done

end=$(date +%s)

runtime=$((end-start))

echo $runtime
