import numpy as np
import random
import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import unittest, time, sys
import models.optimized.densenet_new as densenet_optim
import models.optimized.vnet_new as vnet_optim
import models.optimized.word_language_model_new as wlm_optim
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import logging
import sys
import os
import horovod.torch as hvd
local_rank=int(sys.argv[1])
world_size=int(sys.argv[2])
use_amp=int(sys.argv[3])
if use_amp==0:
    print(use_amp)
    hvd.init()
    local_rank = hvd.local_rank()
    device=hvd.local_rank()
    logfile="./logs/horovod_log{}.txt".format(local_rank)
    logging.basicConfig(format='[%(asctime)s %(filename)s:%(lineno)s] %(message)s', level=logging.INFO,filename=logfile,filemode='w')
if use_amp==1:
    print(local_rank)
    os.environ["MASTER_ADDR"]="127.0.0.1"
    os.environ["MASTER_PORT"]="10100"
    os.environ["NCCL_SOCKET_IFNAME"]="eth0"
    host_addr_full = 'tcp://' + os.environ['MASTER_ADDR']+':'+os.environ['MASTER_PORT']
    torch.distributed.init_process_group(backend='nccl',init_method=host_addr_full,rank=local_rank,world_size=world_size)
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    all_real_device_ids = [int(i) for i in available_gpus.split(",")]
    logfile="./logs/apex_log{}.txt".format(local_rank)
    logging.basicConfig(format='[%(asctime)s %(filename)s:%(lineno)s] %(message)s', level=logging.INFO,filename=logfile,filemode='w')
    if len(all_real_device_ids):
         device_ids = list(range(len(all_real_device_ids)))
         logging.info(f"from CUDA_VISIBLE_DEVICES:{device_ids}")
    device=device_ids[0]


seed=local_rank%1234*100+20 #+np.random.randint(1000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
logging.info(f"seed {seed} local rank: {local_rank}; world size: {world_size}")
print(logfile)
N = 32
chunks = int(sys.argv[4])
total_iters = 2    # (warmup + benchmark)
iterations = 1
z = torch.ones(N, 3, 224, 224, requires_grad=True)
y = torch.rand_like(z,requires_grad=True)*10
x = y+z
target = torch.ones(N).type("torch.LongTensor")

logging.info("inputs norm for rank {} : {}".format(local_rank,x.norm()))
if True:
    logging.info(f"run program {local_rank}")
    if True:
        model = densenet_optim.densenet264()
        model = model.cuda(device)
        optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
        if use_amp==1:
            model, optimizer = amp.initialize(model, optimizer,opt_level='O0')
            model =DDP(model,message_size=1e9,delay_allreduce=True,gradient_average=True,allreduce_always_fp32=True,retain_allreduce_buffers=True)
            torch.distributed.barrier()
            logging.info("apex sync")
        if use_amp==0:
           # horovod
           optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
           hvd.broadcast_parameters(model.state_dict(), root_rank=0)
           hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        # switch the model to train mode
        model.train()
        # convert the model and input to cuda
        input_var = x.cuda(device)
        target_var = target.cuda(device)
        # declare the optimizer and criterion
        criterion = nn.CrossEntropyLoss().cuda(device)
        with cudnn.flags(enabled=True, benchmark=True):
            for i in range(total_iters):
                logging.info(f"local_rank {local_rank}   iteration {i}")
                for j in range(iterations):
                    output = model(input_var, chunks=chunks)
                    loss = criterion(output, target_var)
                    logging.info(f"local_rank {local_rank}   loss   {loss}")
                    logging.info(f"local_rank {local_rank}   loss  requires_grad  {loss.requires_grad}")
                    logging.info(f"local_rank {local_rank}   loss grad_fn  {loss.grad_fn}")
                    optimizer.zero_grad()
                    if use_amp==1:
                        with amp.scale_loss(loss,optimizer,delay_unscale=False) as scaled_loss:
                            scaled_loss.backward() 
                    if use_amp==0:
                        loss.backward()
                    count_param=0
                    norm_total=0
                    for param in model.parameters():
                        if param.requires_grad:
                            count_param=count_param+1
                            norm_total+=param.data.norm()
                    logging.info("rank {} parameter norm {} {}".format(local_rank,count_param,norm_total))
                    optimizer.step()
                torch.cuda.synchronize()
