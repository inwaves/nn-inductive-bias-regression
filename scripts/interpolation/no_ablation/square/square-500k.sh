set -x #echo on
start=$(date +%s)
num_iter=1

for ((i=1;i<=num_iter;i++))
do
  python3 1d_regression.py --tag=square-500k --dataset=square --generalisation_task=interpolation --model_type=ShallowRelu --hidden_units=500000 --learning_rate=0.000002 --adjust_data_linearly=True --early_stopping=True --num_epochs=100000
done

end=$(date +%s)

runtime=$((end-start))

echo $runtime
