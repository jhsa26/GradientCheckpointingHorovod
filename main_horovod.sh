#!/bin/bash
cd /projects/mtl/
code=test_memory_optimized_horovod_apex_comparison.py
world_size=4
chunks=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
use_amp=0
local_rank=0  # here can be any number 
horovodrun -np 4 -H localhost:4 python $code $local_rank $world_size $use_amp $chunks>log_horovod.txt 
