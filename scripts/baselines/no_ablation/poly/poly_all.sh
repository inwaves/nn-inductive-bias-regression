filenames="poly-10.sh poly-100.sh poly-500.sh poly-1k.sh poly-5k.sh poly-10k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/baselines/no_ablation/poly/poly_sub_script.sh $val;
    done
done
