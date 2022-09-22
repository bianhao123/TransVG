CODE_PATH=`pwd`
# docker run --gpus all -v $CODE_PATH:/workspace/ --shm-size="64g" -it djiajun1206/vg:pytorch1.5 /bin/bash
docker run --gpus all -v $CODE_PATH:/workspace/ --shm-size="64g" -it bianhao/tranvg /bin/bash
