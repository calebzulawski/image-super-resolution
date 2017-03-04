#!/bin/bash

# Source file containing
source ./imagenet_credentials.sh

mkdir -p tars images

while read line; do
    wnid=$(echo $line | cut -d" " -f1 -)
    tar=tars/${wnid}.tgz
    images=images/original/${wnid}

    if [ -f "$tar" ]; then
        echo "Skipping $wnid..."
        continue
    fi

    url="http://www.image-net.org/download/synset?wnid=${wnid}&username=${username}&accesskey=${accesskey}&release=latest&src=stanford"

    wget -O $tar $url
    mkdir $images
    tar -C $images -xf $tar
done <wnids.txt
