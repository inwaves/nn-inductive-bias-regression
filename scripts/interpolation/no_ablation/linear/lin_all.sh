filenames="linear-10.sh linear-100.sh linear-500.sh linear-1k.sh linear-5k.sh linear-10k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/interpolation/no_ablation/linear/linear_sub_script.sh $val;
    done
done