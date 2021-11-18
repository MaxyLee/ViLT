set -ex

NUM_GPUS=8
NUM_NODES=1
PTM_DIR=pretrained/vilt_200k_mlm_itm.ckpt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=3335
export NODE_RANK=0

# finetune on VQAv2
VQAv2_DATA_DIR=data/VQAv2

# python run.py with \
#     data_root=$VQAv2_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=$NUM_NODES \
#     task_finetune_vqa_randaug \
#     per_gpu_batchsize=64 \
#     load_path=$PTM_DIR

# finetune on SNLI-VE
VE_DATA_DIR=/data/share/UNITER/origin_imgs/flickr30k

# CUDA_VISIBLE_DEVICES=4 python run.py with \
#     data_root=$VE_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=$NUM_NODES \
#     task_finetune_ve \
#     per_gpu_batchsize=64 \
#     load_path=$PTM_DIR

# finetune on NLVR2
NLVR2_DATA_DIR=data/NLVR2

# python run.py with \
#     data_root=$NLVR2_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=$NUM_NODES \
#     task_finetune_nlvr2_randaug \
#     per_gpu_batchsize=32 \
#     load_path=$PTM_DIR \

# finetune on F30K IR/TR
F30K_DATA_DIR=data/flickr30k

python run.py with \
    data_root=$F30K_DATA_DIR \
    num_gpus=$NUM_GPUS \
    num_nodes=$NUM_NODES \
    task_finetune_irtr_f30k \
    per_gpu_batchsize=4 \
    load_path=$PTM_DIR

# finetune on COCO IR/TR
COCO_DATA_DIR=/data/share/UNITER/origin_imgs/coco_original

# python run.py with \
#     data_root=$COCO_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=$NUM_NODES \
#     task_finetune_irtr_coco_randaug \
#     per_gpu_batchsize=4 \
#     load_path=$PTM_DIR

# finetune on CUB IR/TR
CUB_DATA_DIR=/data/share/data/birds/cub_arrows

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py with \
#     data_root=$CUB_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=$NUM_NODES \
#     task_finetune_irtr_cub_randaug \
#     per_gpu_batchsize=8 \
#     load_path=$PTM_DIR