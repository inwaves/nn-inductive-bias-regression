filenames="chebyshev-10.sh chebyshev-100.sh chebyshev-500.sh chebyshev-1k.sh chebyshev-5k.sh chebyshev-10k.sh"


for val in $filenames;
do
    for ((i=0; i<$1; i++));
    do
    sbatch ./scripts/extrapolation/no_ablation/chebyshev/chebyshev_sub_script.sh $val;
    done
done