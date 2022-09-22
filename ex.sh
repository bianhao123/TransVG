GPUS=$1
array=(${GPUS//,/ })
echo $array
echo ${#array[@]}