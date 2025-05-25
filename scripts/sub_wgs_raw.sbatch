#!/bin/bash
#SBATCH --mem=10g
#SBATCH -p general
#SBATCH -t 5:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=owenjf@live.unc.edu
#SBATCH -J raw_test 


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
        --causal)
            causal="$2"
            shift
            ;;
        --prop)
            prop="$2"
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
echo "n_ldrs: $n_ldrs"
echo "causal: $causal"
echo "prop: $prop"


package=/work/users/o/w/owenjf/image_genetics/methods/package_pub/wgs_simulation.py
main_dir=/work/users/o/w/owenjf/image_genetics/methods/simu_wgs_rel

ldr_prefix=images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal5_heri"$heri"_a"$a"_w"$w"_100voxels_v"$v"_c"$c"

# consider relatedness
## wgs
if [ ! -e  $main_dir/gwas/"$ldr_prefix"_prop"$prop"_raw_100kb ]
then
    mkdir $main_dir/gwas/"$ldr_prefix"_prop"$prop"_raw_100kb
fi

if [ $causal == 'null' ]
then
    chr_start=2
    chr_end=22
else
    chr_start=1
    chr_end=21
fi

for (( chr=$chr_start; chr<=$chr_end; chr+=2 ))
do
    python $package --regions /work/users/o/w/owenjf/image_genetics/methods/bfiles/wgs/"$causal"_regions/mt100k/"$percent"percent/chr"$chr"_"$causal"_regions_"$percent"percent_geno_array.h5 --annot /work/users/o/w/owenjf/image_genetics/methods/bfiles/wgs/"$causal"_regions/mt100k/"$percent"percent/chr"$chr"_"$causal"_regions_"$percent"percent_annot.h5 --null-model /work/users/o/w/owenjf/image_genetics/methods/simu_wgs_rel/gwas/"$ldr_prefix"_null_null_model.h5 --out /work/users/o/w/owenjf/image_genetics/methods/simu_wgs_rel/gwas/"$ldr_prefix"_prop"$prop"_raw_100kb --n-ldrs $n_ldrs --causal $causal 
done
