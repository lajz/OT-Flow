#!/bin/bash
for i in $(seq 1 $2); do
    echo $i
    sbatch sweep.sh $1
done
