#!/bin/bash
#SBATCH --mem=20g
#SBATCH -p general
#SBATCH -t 5:00:00
#SBATCH -c 2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=owenjf@live.unc.edu
#SBATCH -J wgs_preprocess 


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


package=/work/users/o/w/owenjf/image_genetics/methods/package_pub/heig.py
main_dir=/work/users/o/w/owenjf/image_genetics/methods/simu_wgs_rel_0325

ldr_prefix=images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal"$causal"_heri"$heri"_a"$a"_w"$w"_"$dist"_100voxels_v"$v"_c"$c"

## read images
python $package --read-image --image-txt $main_dir/data/"$ldr_prefix".txt  --coord-txt $main_dir/data/images_coord.txt  --out $main_dir/data/"$ldr_prefix"_images
# rm $main_dir/data/"$ldr_prefix".txt

## fpca
python $package --fpca --image $main_dir/data/"$ldr_prefix"_images_images.h5 --all-pc --out $main_dir/data/"$ldr_prefix"_fpca --bw-opt 0.01

## ldr
python $package --make-ldr --image $main_dir/data/"$ldr_prefix"_images_images.h5 --covar $main_dir/covar/ukb_kinship0.05_"$percent"percent_10ksub_pc1.txt --bases $main_dir/data/"$ldr_prefix"_fpca_bases_top100.npy --out $main_dir/data/"$ldr_prefix"_ldr
# rm $main_dir/data/"$ldr_prefix"_images_images.h5 

## null model
python $package --rv-null --out $main_dir/gwas/"$ldr_prefix"_null --bases $main_dir/data/"$ldr_prefix"_fpca_bases_top100.npy --covar $main_dir/covar/ukb_kinship0.05_"$percent"percent_10ksub_pc1.txt --ldrs $main_dir/data/"$ldr_prefix"_ldr_ldr_top100.txt

## relatedness
python $package --relatedness --out $main_dir/gwas/"$ldr_prefix"_rel  --covar $main_dir/covar/ukb_kinship0.05_"$percent"percent_10ksub_pc1.txt --geno-mt /work/users/o/w/owenjf/image_genetics/methods/bfiles/relatedness/ukb_cal_allchr_cleaned_maf_gene_hwe_white_kinship0.05_"$percent"percent_10ksub.mt --grch37 --spark-conf /work/users/o/w/owenjf/image_genetics/methods/package_pub/test_output/relatedness/spark_config_large_mem.json --ldrs $main_dir/data/"$ldr_prefix"_ldr_ldr_top100.txt --threads 2
# rm $main_dir/data/"$ldr_prefix"_ldr_ldr_top100.txt
rm $main_dir/gwas/"$ldr_prefix"_rel_hail.log
