set -x #echo on
start=$(date +%s)
num_iter=1

for ((i=1;i<=num_iter;i++))
do
  python3 1d_regression.py --tag=sq-10 --dataset=square --generalisation_task=baseline --model=ASIShallowRelu --hidden_units=10 --learning_rate=0.25 --adjust_data_linearly=True
done

end=$(date +%s)

runtime=$((end-start))

echo $runtime
