filenames="sine-10.sh sine-100.sh sine-500.sh sine-1k.sh sine-10k.sh sine-5k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
      sbatch ./scripts/baselines/optimiser_ablation/optimiser_sub.sh $val $2;
    done
done