#sh stop_and_remove_docker.sh
image="harbor.cloudwalk.work/aiflow/nvcr.io/nvidia/pytorch:HVD2"
work_path=/home/yckj2766/pytorch_memonger_new/
train_exe=main.sh
docker run -ti --network=host \
-v ${work_path}:/projects/mtl \
-v /data:/data \
--ipc=host \
${image}   /bin/bash
#--entrypoint "/bin/sh"  ${image} /projects/mtl/${train_exe}
