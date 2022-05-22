set -x #echo on
start=$(date +%s)
num_iter=1

for ((i=1;i<=num_iter;i++))
do
  python3 1d_regression.py --tag=sq-cos-noes100k --num_datapoints=10 --early_stopping=False --num_epochs=100000 --lr_schedule=cosine --dataset=square --generalisation_task=baseline --model_type=ASIShallowRelu --hidden_units=100000 --learning_rate=0.00001 --adjust_data_linearly=True
done

end=$(date +%s)

runtime=$((end-start))

echo $runtime
