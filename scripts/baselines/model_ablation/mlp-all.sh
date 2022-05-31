filenames="mlp-10.sh mlp-100.sh mlp-150.sh mlp-500.sh"

for val in $filenames;
do
    for dataset in $2;
    do
      for ((i=0; i<$1; i++));
      do
      sbatch ./scripts/baselines/model_ablation/mlp-sub-script.sh $val $dataset;
      done
    done
done