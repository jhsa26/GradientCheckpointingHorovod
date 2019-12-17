# run this script, then enter /projects/mtl and bash main.sh

sh stop_and_remove_docker.sh          # stop all containers
image="nvcr.io/nvidia/pytorch:HVD2"   # use your image
work_path=/pytorch_memonger-master/  # use your work path
train_exe=main.sh
docker run -ti --network=host \
-v ${work_path}:/projects/mtl \
-v /data:/data \
--ipc=host \
${image}   /bin/bash
#--entrypoint "/bin/sh"  ${image} /projects/mtl/${train_exe}
