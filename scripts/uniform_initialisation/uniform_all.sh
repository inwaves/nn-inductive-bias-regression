filenames="sin_var_10.sh sin_var_160.sh sin_var_2560.sh sin_var_10240.sh sin_var_50k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
      sbatch ./scripts/uniform_initialisation/uniform_sub.sh $val 2 1
      sbatch ./scripts/uniform_initialisation/uniform_sub.sh $val 1 5;
    done
done