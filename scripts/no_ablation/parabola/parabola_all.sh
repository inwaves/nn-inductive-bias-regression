filenames="parabola-10.sh parabola-100.sh parabola-500.sh parabola-1k.sh parabola-5k.sh parabola-10k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/no_ablation/parabola/parabola_sub_script.sh $val;
    done
done