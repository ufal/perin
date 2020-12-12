#!/bin/bash

if [ -z "$1" ]; then
    qstat -u '*'
    echo -n "Select node: "
    read -rsn1 node
else
    node=$1
fi

if [ "$1" == "jupyter" ]; then
    qsub -N JUPYTER -cwd -pe smp 2 -l gpu=1,gpu_ram=11G,mem_free=25G,act_mem_free=25G,h_data=25G -q "gpu*" -j y scripts/jupyter.sh
elif [ "$1" == "preprocess" ]; then
    qsub -N PREPROSESS -cwd -pe smp 4 -l mem_free=48G,act_mem_free=48G -q "cpu*" -j y scripts/run.sh
elif [ "$1" == "evaluate" ]; then
    qsub -N EVALUERING -cwd -pe smp 1 -l mem_free=32G,act_mem_free=32G,h_data=32G -q "cpu*" -j y scripts/validation.sh "$2" 0 "$3" "$4" "$5"
elif [ "$1" == "predict" ]; then
    qsub -N PREDIKSJON -cwd -pe smp 2 -l gpu=1,gpu_ram=11G,mem_free=25G,act_mem_free=25G,h_data=25G -q "gpu*" -j y scripts/predict.sh
elif [ "$1" == "single" ]; then
    qsub -N PERIN -cwd -pe smp 2 -l gpu=1,gpu_ram=11G,mem_free=25G,act_mem_free=25G,h_data=25G -q "gpu*" -j y scripts/run.sh
else
    if [ "$1" == "auto" ] || [ "$1" == "double" ] ; then
        node="gpu*"
    else
        node="gpu-ms.q@gpu-node${node}"
    fi
    qsub -N PERIN -cwd -pe smp 4 -l gpu=2,gpu_ram=11G,mem_free=50G,act_mem_free=50G,h_data=50G -q "${node}" -j y scripts/run.sh
fi

watch "qstat -u '*' | tail -n 12"
