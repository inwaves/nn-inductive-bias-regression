set -x #echo on
start=$(date +%s)
num_iter=1

for ((i=1;i<=num_iter;i++))
do
  python3 1d_regression.py --tag=lin-unadjusted-500 --early_stopping=True --num_epochs=100000 --lr_schedule=plateau --dataset=linear --generalisation_task=interpolation --model_type=ShallowRelu --hidden_units=500 --learning_rate=0.005 --adjust_data_linearly=False
done

end=$(date +%s)

runtime=$((end-start))

echo $runtime
