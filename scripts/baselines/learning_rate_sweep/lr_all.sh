lr="0.01 0.001 0.0005 0.0001 0.00005 0.00001 0.000005"


for val in $lr;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/baselines/learning_rate_sweep/lr_sub_script.sh $val;
    done
done