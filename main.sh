#!/bin/bash
cd /projects/mtl/
code=test_memory_optimized_horovod_apex_comparison.py
world_size=3
chunks=4
export CUDA_VISIBLE_DEVICES=3,4,5
horovodrun -np 3 -H localhost:3 python $code  0 $world_size $chunks>log.txt 






<<block
exit
export CUDA_VISIBLE_DEVICES=3
nohup  python test_memory_optimized.py  0 $world_size $chunks >log1.txt &
export CUDA_VISIBLE_DEVICES=4   
nohup python test_memory_optimized.py  1 $world_size  $chunks >log2.txt &
#nohup python test_memory_optimized.py  1 $world_size  >log2.txt &
export CUDA_VISIBLE_DEVICES=5
# python test_memory_optimized.py  2 $world_size  #>log3.txt  &
nohup python test_memory_optimized.py  2 $world_size  $chunks >log3.txt  &
tail -f log1.txt
exit
block
