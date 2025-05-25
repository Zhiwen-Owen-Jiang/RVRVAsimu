conda activate hail

percent=20
heri=0.01
w=0.8
n_ldrs=15
prop=90
a=1.8
v=1


for heri in {0.01,0.05}
do
    for w in {1.0,0.8}
    do
        ## simulating data
        # sbatch -t 1:00:00 --mem=10g -p general --wrap="python /work/users/o/w/owenjf/image_genetics/methods/code/relatedness_images_wgs.py --heri $heri -w $w -v $v --percent $percent -c 1:100" -J generate_data --mail-type=END,FAIL --mail-user=owenjf@live.unc.edu
        for c in {1..100}
        do
            if [ ! -e gwas/images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal5_heri"$heri"_a"$a"_w"$w"_100voxels_v1_c"$c"_rel_ldr_loco_preds.h5 ]
            then
                ## preprocessing
                echo gwas/images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal5_heri"$heri"_a"$a"_w"$w"_100voxels_v1_c"$c"_rel_ldr_loco_preds.h5
                sbatch preprocess_steps.sbatch --percent $percent --heri $heri -c $c -w $w -v $v -a $a
            fi
        done
    done
done


n_ldrs=15 # 8
prop=90 # 80
for heri in {0.01,0.05}
do
    for w in {1.0,0.8}
    do
        for causal in {causal,null}
        do
            for c in {1..100}
            do
                sbatch sub_wgs_raw.sbatch --heri $heri -a 1.8 -w $w --percent $percent -v $v -c $c --n-ldrs $n_ldrs --causal $causal --prop $prop 
                # sbatch sub_wgs_rel.sbatch --heri $heri -a 1.8 -w $w --percent $percent -v $v -c $c --n-ldrs $n_ldrs --causal $causal --prop $prop
            done
        done
    done
done


for prop in {80,90,100}
do
    for heri in {0.01,0.05}
    do
        for w in {0.8,1.0}
        do
            for adj in {rel,raw}
            do
               python summarize_rel_count.py --prefix images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal5_heri"$heri"_a1.8_w"$w"_100voxels_v1 --prop "$prop" --rel "$adj"
            done
        done
    done
done



# 0325 for debug
# n_ldrs=70, corr=95; n_ldrs=50, corr=90
sh generate_images.sh --heri 0.05 -w 0.8 -v 1 --percent 20 -c 1 --causal 1 --dist skewed
sh generate_images.sh --heri 0.2 -w 0.8 -v 1 --percent 20 -c 1 --causal 1 --dist normal
sbatch preprocess_steps.sbatch --heri 0.05 -a 1.8 -w 0.8 --percent 0 -v 1 -c 1 --causal 1 --dist skewed
sbatch preprocess_steps.sbatch --heri 0.2 -a 1.8 -w 0.8 --percent 0 -v 1 -c 1 --causal 02 --dist normal
sbatch permutation.sbatch --heri 0.05 -a 1.8 -w 0.8 --percent 0 -v 1 -c 1 --causal 1 --dist skewed --n-ldrs 70 --prop 95
sbatch permutation.sbatch --heri 0.2 -a 1.8 -w 0.8 --percent 0 -v 1 -c 1 --causal 02 --dist normal --n-ldrs 70 --prop 95
sbatch sub_wgs_rel_null.sbatch --heri 0.05 -a 1.8 -w 0.8 --percent 0 -v 1 -c 1 --causal 1 --dist skewed --n-ldrs 70 --prop 95
sbatch sub_wgs_rel_causal.sbatch --heri 0.2 -a 1.8 -w 0.8 --percent 0 -v 1 -c 1 --causal 02 --dist normal --n-ldrs 70 --prop 95


# 0325 for final results
## data generation
for percent in {0,20,60}
do
    for causal in {1,02}
    do
        for heri in {0.05,0.1}
        do
            for dist in {normal,skewed}
            do
                if [ ! -e data/images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal"$causal"_heri"$heri"_a1.8_w0.8_"$dist"_100voxels_v1_c1.txt ]
                then
                    sh generate_images.sh --heri $heri -w 0.8 -v 1 --percent $percent -c 1 --causal $causal --dist $dist
                else
                    echo "$percent"percent_causal"$causal"_heri"$heri"_"$dist"
                fi
            done
        done
    done
done

## preprocess
for percent in {0,20,60}
do
    for causal in {1,02}
    do
        for heri in {0.05,0.1}
        do
            for dist in {normal,skewed}
            do
                if [ ! -e gwas/images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal"$causal"_heri"$heri"_a1.8_w0.8_"$dist"_100voxels_v1_c1_rel_ldr_loco_preds.h5 ]
                then
                    sbatch preprocess_steps.sbatch --heri $heri -a 1.8 -w 0.8 --percent $percent -v 1 -c 1 --causal $causal --dist $dist
                else
                    echo "$percent"percent_causal"$causal"_heri"$heri"_"$dist"
                fi
            done
        done
    done
done

## permutation
for percent in {0,20,60}
do
    for causal in {1,02}
    do
        for heri in {0.05,0.1}
        do
            for dist in {normal,skewed}
            do
                for prop in {90,95}
                do
                    if [ $prop == 90 ]
                    then
                        n_ldrs=50
                    else
                        n_ldrs=70
                    fi

                    if [ ! -e gwas/images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal"$causal"_heri"$heri"_a1.8_w0.8_"$dist"_100voxels_v1_c1_prop"$prop"_permute_2e7_burden_perm.h5 ]
                    then
                        sbatch permutation.sbatch --heri $heri -a 1.8 -w 0.8 --percent $percent -v 1 -c 1 --causal $causal --dist $dist --n-ldrs $n_ldrs --prop $prop
                    else
                        echo "$percent"percent_causal"$causal"_heri"$heri"_"$dist"_prop"$prop"
                    fi
                done
            done
        done
    done
done

## analysis
for percent in {0,20,60}
do
    for causal in {1,02}
    do
        for heri in {0.05,0.1}
        do
            for dist in {normal,skewed}
            do
                for prop in {90,95}
                do
                    if [ $prop == 90 ]
                    then
                        n_ldrs=50
                    else
                        n_ldrs=70
                    fi
                    
                    for i in {1..50}
                    do
                        sbatch sub_wgs_rel_null.sbatch --heri $heri -a 1.8 -w 0.8 --percent $percent -v 1 -c 1 --causal $causal --dist $dist --n-ldrs $n_ldrs --prop $prop
                        sbatch sub_wgs_rel_causal.sbatch --heri $heri -a 1.8 -w 0.8 --percent $percent -v 1 -c 1 --causal $causal --dist $dist --n-ldrs $n_ldrs --prop $prop
                    done
                done
            done
        done
    done
done



## permutation w/o doing sample relatedness adjustment
for percent in {0,20,60}
do
    for causal in {1,02}
    do
        for heri in {0.05,0.1}
        do
            for dist in {normal,skewed}
            do
                for prop in {90,95}
                do
                    if [ $prop == 90 ]
                    then
                        n_ldrs=50
                    else
                        n_ldrs=70
                    fi

                    if [ ! -e gwas/images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal"$causal"_heri"$heri"_a1.8_w0.8_"$dist"_100voxels_v1_c1_prop"$prop"_permute_2e7_raw_burden_perm.h5 ]
                    then
                        sbatch permutation_raw.sbatch --heri $heri -a 1.8 -w 0.8 --percent $percent -v 1 -c 1 --causal $causal --dist $dist --n-ldrs $n_ldrs --prop $prop
                    else
                        echo "$percent"percent_causal"$causal"_heri"$heri"_"$dist"_prop"$prop"
                    fi
                done
            done
        done
    done
done


## analysis w/o doing sample relatedness adjustment for null
for percent in {0,20,60}
do
    for causal in {1,02}
    do
        for heri in {0.05,0.1}
        do
            for dist in {normal,skewed}
            do
                for prop in {90,95}
                do
                    if [ $prop == 90 ]
                    then
                        n_ldrs=50
                    else
                        n_ldrs=70
                    fi

                    for i in {1..50}
                    do
                        sbatch sub_wgs_raw_null.sbatch --heri $heri -a 1.8 -w 0.8 --percent $percent -v 1 -c 1 --causal $causal --dist $dist --n-ldrs $n_ldrs --prop $prop
                    done
                done
            done
        done
    done
done


## analysis w/o doing sample relatedness adjustment for causal
percent=0
for causal in {1,02}
do
    for heri in {0.05,0.1}
    do
        for dist in {normal,skewed}
        do
            for prop in {90,95}
            do
                if [ $prop == 90 ]
                then
                    n_ldrs=50
                else
                    n_ldrs=70
                fi

                for i in {1..50}
                do
                    sbatch sub_wgs_raw_causal.sbatch --heri $heri -a 1.8 -w 0.8 --percent $percent -v 1 -c 1 --causal $causal --dist $dist --n-ldrs $n_ldrs --prop $prop
                done
            done
        done
    done
done
