#!/bin/bash
TIME=$(date "+%Y-%m-%d-%H-%M-%S")


OUTPUT_PATH=./outputs
TRAIN_LIST=/home/lxztju/pytorch_classification/sample_files/imgs/listfile.txt
VAL_LIST=/home/lxztju/pytorch_classification/sample_files/imgs/listfile.txt

#train
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 -u -m torch.distributed.launch --nproc_per_node 4  ./tools/train_val.py \
#     --model_name=resnet18 \
#     --lr  0.01 --epochs 70  --batch-size 128  -j 4 \
#     --output=$OUTPUT_PATH/$TIME \
#     --train_list=$TRAIN_LIST \
#     --val_list=$VAL_LIST \
#     --num_classes=2 \
#     --is_pretrained
    
    
    
    
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 -u -m torch.distributed.launch --nproc_per_node 1  ./tools/evaluation.py \
#     --model_name=resnet18 \
#     --batch-size 64  -j 4 \
#     --output=$OUTPUT_PATH/$TIME \
#     --val_list=$VAL_LIST \
#     --tune_from='/home/lxztju/pytorch_classification/ouputs/xxx/epoch_4.pth' \
#     --num_classes=2
    

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python3 -u -m torch.distributed.launch --nproc_per_node 1  ./tools/predict.py \
#     --model_name=resnet18 \
#     --batch-size 64  -j 4 \
#     --output=$OUTPUT_PATH/$TIME \
#     --val_list=$VAL_LIST \
#     --tune_from='/home/lxztju/pytorch_classification/ouputs/xxx/epoch_4.pth' \
#     --num_classes=2
