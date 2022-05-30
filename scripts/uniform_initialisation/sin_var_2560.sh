set -x #echo on
start=$(date +%s)
num_iter=1

for ((i=1;i<=num_iter;i++))
do
  python3 1d_regression.py --tag=uniforminit --a_w="$1" --a_b="$2" --dataset=sine --generalisation_task=baseline --model_type=ASIShallowRelu --hidden_units=2560 --learning_rate=0.001 --adjust_data_linearly=True --early_stopping=True --num_epochs=100000 --init=normal
done

end=$(date +%s)

runtime=$((end-start))

echo $runtime