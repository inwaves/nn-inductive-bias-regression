filenames="sine-10.sh sine-100.sh sine-500.sh sine-1k.sh sine-5k.sh sine-10k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/interpolation/no_ablation/sine/sine_sub_script.sh $val;
    done
done