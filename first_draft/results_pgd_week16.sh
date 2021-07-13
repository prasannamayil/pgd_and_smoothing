dir=/cluster/scratch/pmayilvah/Week16_pgd_results

for i in {1..11}
do
    dir2=$dir/$i
    mkdir -p $dir2
    for j in {1..10}
    do
	bsub -n 1 -W 04:00 -R "rusage[mem=20480,scratch=20480,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -J res_pgd python3 results_pgd_week16.py $i $j

    done
done
