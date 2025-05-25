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
echo "a: 1.8"
echo "w: $w"
echo "percent: $percent"
echo "v: $v"
echo "c: $c"
echo "causal: $causal"
echo "distribution: $dist"


package=/work/users/o/w/owenjf/image_genetics/methods/simu_package/relatedness_images_wgs.py
main_dir=/work/users/o/w/owenjf/image_genetics/methods/simu_wgs_rel_0325

ldr_prefix=images_rare_snps_kinship0.05_"$percent"percent_10ksub_causal"$causal"_heri"$heri"_a"$a"_w"$w"_"$dist"_100voxels_v"$v"_c"$c"

if [ $dist == 'skewed' ]
then
    python $package --heri $heri -w $w -v $v --percent $percent -c $c --causal $causal --skewed
else
    python $package --heri $heri -w $w -v $v --percent $percent -c $c --causal $causal 
fi
