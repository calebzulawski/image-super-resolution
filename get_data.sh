#!/bin/bash

# Source file containing
source ./imagenet_credentials.sh

# Size to crop images to
crop=256

# Set up data directories
mkdir -p tars images/original images/crop_$crop/

# Download images
while read line; do
    wnid=$(echo $line | cut -d" " -f1 -)
    tar=tars/${wnid}.tgz
    images=images/original/${wnid}

    if [ -f "$tar" ]; then
        echo "Already downloaded $wnid..."
        continue
    fi

    url="http://www.image-net.org/download/synset?wnid=${wnid}&username=${username}&accesskey=${accesskey}&release=latest&src=stanford"

    wget -O $tar $url
    mkdir $images
    tar -C $images -xf $tar
done <wnids.txt

# Crop images
for dir in images/original/*; do
    wnid=$(basename $dir)

    outdir=images/crop_$crop/$wnid
    if [ -d "$outdir" ]; then
        echo "Already cropped $wnid..."
        continue
    fi

    echo "Cropping $wnid..."
    mkdir $outdir

    for file in $dir/*; do
        mindim=$(identify -format "%h\n%w" $file | sort -n | head -n1)
        if [ "$mindim" -ge "$crop" ]; then
            convert $file -gravity center -crop ${crop}x${crop}+0+0 $outdir/$(basename $file)
        fi
    done
done

#split into train, test and validation sets
setpath=images/sets

if [ ! -d "$setpath" ];
then
    mkdir -p $setpath $setpath/test $setpath/validation $setpath/train
    for dir in images/crop_$crop/*
    do
        for image in $dir/*
        do
            set=$(( $RANDOM % 100 ))

            if [ $set -gt 90 ];
            then
                setname='test'
            elif [ $set -gt 80 ];
            then
                setname='validation'
            else
                setname='train'
            fi

            cp $image $setpath/$setname;
        done
    done
else
    echo "already generated dataset"
fi
