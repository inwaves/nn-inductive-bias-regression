filenames="square_var_10.sh square_var_100.sh square_var_1000.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/more_data/md_sub.sh $val;
    done
done