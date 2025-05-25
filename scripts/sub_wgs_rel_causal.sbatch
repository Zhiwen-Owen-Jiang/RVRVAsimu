#!/bin/bash
#SBATCH --mem=10g
#SBATCH -p general
#SBATCH -t 5:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=owenjf@live.unc.edu
#SBATCH -J rel_causal 


while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --heri)
            heri="$2"
            shift # Shift past argument value
            ;;
        -a)
            a="$2"
            shift
            ;;
        -w)
            w="$2"
            shift
            ;;
        --percent)
            percent="$2"
            shift
            ;;
        --causal)
            causal="$2"
            shift
            ;;
        -v)
            v="$2"
            shift
            ;;
        -c)
            c="$2"
            shift
            ;;
        --n-ldrs)
            n_ldrs="$2"
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

package=/work/users/o/w/owenjf/image_genetics/methods/simu_package/run_simulation.py
main_dir=/work/users/o/w/owenjf/image_genetics/methods/simu_wgs_rel_0325

ldr_prefix=images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal"$causal"_heri"$heri"_a"$a"_w"$w"_"$dist"_100voxels_v"$v"_c"$c"

# consider relatedness
python $package --null-model $main_dir/gwas/"$ldr_prefix"_null_null_model.h5 --loco-preds $main_dir/gwas/"$ldr_prefix"_rel_ldr_loco_preds.h5 --out $main_dir/results/causal/"$ldr_prefix"_prop"$prop"_rel_causal --n-ldrs $n_ldrs --sparse-genotype /work/users/o/w/owenjf/image_genetics/methods/bfiles/wgs_0325/"$percent"percent/ukb_chr@_rare_snps_kinship0.05_"$percent"percent_10ksub_merged.npz --perm $main_dir/gwas/"$ldr_prefix"_prop"$prop"_permute_2e7_burden_perm.h5 --causal-idx /work/users/o/w/owenjf/image_genetics/methods/bfiles/wgs_0325/"$percent"percent/ukb_chr@_rare_snps_causal"$causal"_kinship0.05_"$percent"percent_10ksub_merged_causal_index.txt
