filenames="constant-10.sh constant-100.sh constant-500.sh constant-1k.sh constant-5k.sh constant-10k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/baselines/no_ablation/constant/constant_sub_script.sh $val;
    done
done