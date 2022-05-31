filenames="square-10.sh square-100.sh square-500.sh square-1k.sh square-5k.sh square-10k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/early_stopping_ablation/square/sq_sub_script.sh $val $2;
    done
done