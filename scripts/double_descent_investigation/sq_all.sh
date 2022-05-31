filenames="square-1k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/early_stopping_ablation/square/sq_sub_script.sh $val;
    done
done