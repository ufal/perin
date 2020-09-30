#!/bin/bash

MTOOL=$(dirname $0)/../mtool/main.py
for (( i=0; i<$1; i++ ))
do
    $MTOOL --i $i --strings --pretty --read mrp --score mrp --framework $4 --gold "$5" --errors "sample.dot" --write dot "$2" > /dev/null 2>&1
    sed -i -e 's/<font face="Courier"><\/font>//g' "sample.dot"
    dot -Tpng "sample.dot" > "$3$i.png"
    rm "sample.dot"
done
