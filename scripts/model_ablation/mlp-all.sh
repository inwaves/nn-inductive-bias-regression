filenames="mlp-10.sh mlp-50.sh mlp-100.sh mlp-150.sh"

for val in $filenames;
do
    for dataset in $2;
    do
      for ((i=0; i<$1; i++));
      do
      sbatch ./scripts/model_ablation/mlp-sub-script.sh $val $dataset;
      done
    done
done