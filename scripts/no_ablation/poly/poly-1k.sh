set -x #echo on
start=$(date +%s)
num_iter=1

for ((i=1;i<=num_iter;i++))
do
  python3 1d_regression.py --tag=poly-no-es-1k --early_stopping=True --num_epochs=100000 --lr_schedule=plateau --dataset=polynomial_spline --generalisation_task=baseline --model_type=ShallowRelu --hidden_units=1000 --learning_rate=0.001 --adjust_data_linearly=True
done

end=$(date +%s)

runtime=$((end-start))

echo $runtime
