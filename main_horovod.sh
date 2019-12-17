#!/bin/bash
cd /projects/mtl/
code=test_memory_optimized_horovod_apex_comparison.py
world_size=3
chunks=4
export CUDA_VISIBLE_DEVICES=0,1,2
use_amp=0
local_rank=0  # here can be any number 
horovodrun -np 3 -H localhost:3 python $code $local_rank $world_size $use_amp $chunks>log_horovod.txt 
