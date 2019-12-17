# GradientCheckpointingHorovod

## Introduction

Bert, Roberta, and Xlnet models are too large to be trained. They are often partioned on several GPUs (one defines this as `Model Parallelism`). Actually, layers are dependent which means current layer needs the ouput of preceding layer to continue some calculation. Assuming a large model is splitted into 4 parts each on a single GPU, other parts are waitting for others when one of part is runing some computation. Therefore it does not fully utilize GPU resources. It is not real parallel computing. Here we adopt `Data Parallelism` to train a large model. To do that, We should put a large model on a single GPU and train it with large batch size. Gradient checkpointing technique [(Chen et al., 2016)](https://arxiv.org/abs/1604.06174) is good way to save memory cost. It just saves a few of intermediate variables in forward process. For backward process, it can use saved intermediate variables to recompute dropped intermediate variables needed by the backward process. It just trades slight computation for memory. It is implemented in Pytorch. You can import `checkpoint`, `checkpoint_sequential` function from torch.

~~~python
from torch.utils.checkpoint import checkpoint_sequential,checkpoint
~~~

There are some distributed training libraries such as `DataDistributedParallel` of Pytorch, `Apex` of Nvidia, and `Horovod` of Uber. Both `DataDistributedParallel` and `Apex` are not compatible with `checkpoint_sequential` associated with chunks larger than 1, which means that the computed local gradients are not synchronous for all ranks when performing Ring-allreduce operation. It suprised me. So far, we cannot fix this bug. We will show this bug by through a a comparison of `Horovod` and `Apex` over [a simple network](https://github.com/prigoyal/pytorch_memonger/tree/master/).

## Notes about checkpoint_sequential 
Please note that the `checkpoint_sequential` only works out in case of that the output of current layer is the input of next layer. You should make sure the number of elements in `forward` API is equal to the number of elements of `return` after some calculation.

~~~python
Class balba(torch.nn.Module):
    balabalabala
    def forward(self,A,B,C):
        balabala
        return A,B,C
~~~

If B,C don't require gradients, we let B and C as global variables. Then define your model like this
 
~~~python
Class balba(torch.nn.Module):
    balabalabala
    def forward(self,A):
        some calculation in terms of B,C
        return A
~~~


## How to run
You can use docker to run this test by 

```bash
bash run_docker_checkpoint_sequential.sh
```

or run it directly by

```bash
bash main.sh
```

You can modify `use_amp` in `./test_memory_optimized_horovod_apex_comparison.py` to use or not use `Apex`.
