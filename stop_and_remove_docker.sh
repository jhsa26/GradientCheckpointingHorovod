a=`docker ps -a | grep pytorch | awk '{print $1}'`
docker stop $a
docker rm $a
