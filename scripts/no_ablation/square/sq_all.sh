filenames="square_var_10.sh square_var_160.sh square_var_640.sh square_var_2560.sh square_var_10240.sh square_var_50k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/no_ablation/square/sq_sub_script.sh $val;
    done
done