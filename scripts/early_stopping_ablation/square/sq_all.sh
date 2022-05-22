filenames="sq-10.sh sq-100.sh sq-1k.sh sq-10k.sh sq-100k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/no_ablation/sq/sq_sub_script.sh $val;
    done
done