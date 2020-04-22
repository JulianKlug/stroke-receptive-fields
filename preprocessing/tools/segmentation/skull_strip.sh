#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -i|--input)
    img="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


#Tresholding Image to 0-100
intensity=0.01
outfile="betted_$img"
tmpfile=`mktemp`

echo 'Betting : '${img}

# Thresholding Image to 0-100
fslmaths "${img}" -thr 0.000000 -uthr 100.000000  "${outfile}"
# Creating 0 - 100 mask to remask after filling
fslmaths "${outfile}"  -bin   "${tmpfile}";
fslmaths "${tmpfile}.nii.gz" -bin -fillh "${tmpfile}"
# Presmoothing image
fslmaths "${outfile}"  -s 1 "${outfile}";
# Remasking Smoothed Image
fslmaths "${outfile}" -mas "${tmpfile}"  "${outfile}"
# Running bet2
bet2 "${outfile}" "${outfile}" -f ${intensity} -v
# Using fslfill to fill in any holes in mask
fslmaths "${outfile}" -bin -fillh "${outfile}_Mask"
# Using the filled mask to mask original image
fslmaths "${img}" -mas "${outfile}_Mask"  "${outfile}"


# ######################
# ## If no pre-smoothing
# ######################
# outfile_nosmooth="Head_Image_1_SS_0.01_nopresmooth"
# fslmaths "$img" -thr 0.000000 -uthr 100.000000  "${outfile_nosmooth}"
# # Creating 0 - 100 mask to remask after filling
# fslmaths "${outfile_nosmooth}"  -bin   "${tmpfile}";
# fslmaths "${tmpfile}" -bin -fillh "${tmpfile}"
# # Running bet2
# bet2 "${outfile_nosmooth}" "${outfile_nosmooth}" -f ${intensity} -v
# # Using fslfill to fill in any holes in mask
# fslmaths "${outfile_nosmooth}" -bin -fillh "${outfile_nosmooth}_Mask"
# # Using the filled mask to mask original image
# fslmaths "$img" -mas "${outfile_nosmooth}_Mask"  "${outfile_nosmooth}"
