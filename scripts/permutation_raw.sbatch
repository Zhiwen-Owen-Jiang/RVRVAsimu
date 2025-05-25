#!/bin/bash
#SBATCH --mem=4g
#SBATCH -p general
#SBATCH -t 2:00:00
#SBATCH -c 1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=owenjf@live.unc.edu
#SBATCH -J permutation 


# this is the first step

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --percent)
            percent="$2"
            shift # Shift past argument value
            ;;
        --causal)
            causal="$2"
            shift # Shift past argument value
            ;;
        --heri)
            heri="$2"
            shift # Shift past argument value
            ;;
        --n-ldrs)
            n_ldrs="$2"
            shift # Shift past argument value
            ;;
        -c)
            c="$2"
            shift
            ;;
        -w)
            w="$2"
            shift
            ;;
        -v)
            v="$2"
            shift
            ;;
        -a)
            a="$2"
            shift
            ;;
        --prop)
            prop="$2"
            shift
            ;;
        --dist)
            dist="$2"
            shift
            ;;
        *)
            echo "Unknown option: $key"
            ;;
    esac
    shift # Shift past the argument key
done

# Print parsed values
echo "heri: $heri"
echo "a: $a"
echo "w: $w"
echo "percent: $percent"
echo "v: $v"
echo "c: $c"
echo "causal: $causal"
echo "distribution: $dist"
echo "n_ldrs: $n_ldrs"
echo "prop: $prop"


package=/work/users/o/w/owenjf/image_genetics/methods/package_pub/heig.py
main_dir=/work/users/o/w/owenjf/image_genetics/methods/simu_wgs_rel_0325

ldr_prefix=images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal"$causal"_heri"$heri"_a"$a"_w"$w"_"$dist"_100voxels_v"$v"_c"$c"

## permutation
if [ $percent == 0 ]
then
    python $package --permute --out $main_dir/gwas/"$ldr_prefix"_prop"$prop"_permute_2e7_raw --n-ldrs $n_ldrs --spark-conf /work/users/o/w/owenjf/image_genetics/methods/package_pub/test_output/rv_sumstats/spark_config_timeout.json --null-model $main_dir/gwas/"$ldr_prefix"_null_null_model.h5 --sparse-genotype /work/users/o/w/owenjf/image_genetics/methods/real_data_analysis/wes/sparse_genotype_phase1to6/ukb23150_c1_b0_v1_phase1to6_hwe_maf_qc  --n-bootstrap 20000000
else
    python $package --permute --out $main_dir/gwas/"$ldr_prefix"_prop"$prop"_permute_2e7_raw --n-ldrs $n_ldrs --spark-conf /work/users/o/w/owenjf/image_genetics/methods/package_pub/test_output/rv_sumstats/spark_config_timeout.json  --null-model $main_dir/gwas/"$ldr_prefix"_null_null_model.h5 --sparse-genotype /work/users/o/w/owenjf/image_genetics/methods/bfiles/wgs_0325/sparse_genotype/ukb23150_c1_b0_v1_percent20_60_hwe_maf_qc  --n-bootstrap 20000000
fi
