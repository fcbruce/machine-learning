#!/bin/bash
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Sat 27 Feb 2016 04:50:10 AM CST
#
#

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done
