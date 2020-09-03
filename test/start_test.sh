#!/bin/bash

size=4
for i in $(seq 0 $(($size-1)))
do
    OMPI_COMM_WORLD_RANK=$i OMPI_COMM_WORLD_SIZE=$size python $@ &
    processes[${i}]=$!
done

for p in ${processes[*]}; do
    wait $p
done