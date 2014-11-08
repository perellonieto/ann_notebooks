#!/bin/bash

for f in `ls ./*.png`
do
    echo "Croping file $f"
    convert $f -trim +repage $f
done

convert ./*.png -set delay 100 new_gif.gif

