filenames="square_var_10.sh square_var_100.sh square_var_500.sh square_var_1k.sh square_var_5k.sh square_var_10k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/no_ablation/square/sq_sub_script.sh $val;
    done
done