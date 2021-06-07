#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

#!/bin/bash

size=4
for i in $(seq 0 $(($size-1)))
do
    OMPI_COMM_WORLD_LOCAL_RANK=$i OMPI_COMM_WORLD_RANK=$i OMPI_COMM_WORLD_SIZE=$size python $@ &
    processes[${i}]=$!
done

for p in ${processes[*]}; do
    wait $p
done
