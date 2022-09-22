GPUS=$1
sh train_dataset.sh $GPUS refcoco+
sh train_dataset.sh $GPUS refcocog_g
sh train_dataset.sh $GPUS refcocog_u