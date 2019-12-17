#!/bin/bash
cd /projects/mtl/
rm -rf nohup.out
code=test_memory_optimized_horovod_apex_comparison.py
world_size=4
chunks=4
export CUDA_VISIBLE_DEVICES=0
local_rank=0; use_amp=1
nohup python $code   $local_rank  $world_size $use_amp  $chunks >nohupout_apex1 2>&1 & 
export CUDA_VISIBLE_DEVICES=1
local_rank=1; use_amp=1
nohup python $code   $local_rank  $world_size $use_amp  $chunks >nohupout_apex2 2>&1 & 
export CUDA_VISIBLE_DEVICES=2
local_rank=2; use_amp=1  # use_amp=0 or 1, 0: not use amp; 1:  use amp
nohup python $code   $local_rank  $world_size $use_amp  $chunks >nohupout_apex3 2>&1 &
export CUDA_VISIBLE_DEVICES=3
local_rank=3; use_amp=1  # use_amp=0 or 1, 0: not use amp; 1:  use amp
nohup python $code   $local_rank  $world_size $use_amp  $chunks >nohupout_apex4 2>&1 &
