dir=$(dirname "$1")
filename=$(basename "$1" ".mrp")
train_filename="${dir}/${filename}_train.mrp"
val_filename="${dir}/${filename}_val.mrp"

awk -v train="$train_filename" -v test="$val_filename" '{if(rand()<0.9) {print > train} else {print > test}}' "$1"
