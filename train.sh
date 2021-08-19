set -ex

NUM_GPUS=1
NUM_NODES=1
PTM_DIR=/data/share/ViLT/pretrained/vilt_200k_mlm_itm.ckpt

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=2334
export NODE_RANK=0

# finetune on VQAv2
VQAv2_DATA_DIR=/data/share/UNITER/origin_imgs/coco_original

# python run.py with \
#     data_root=$VQAv2_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=$NUM_NODES \
#     task_finetune_vqa_randaug \
#     per_gpu_batchsize=64 \
#     load_path=$PTM_DIR

# finetune on SNLI-VE
VE_DATA_DIR=/data/share/UNITER/origin_imgs/flickr30k

CUDA_VISIBLE_DEVICES=4 python run.py with \
    data_root=$VE_DATA_DIR \
    num_gpus=$NUM_GPUS \
    num_nodes=$NUM_NODES \
    task_finetune_ve \
    per_gpu_batchsize=64 \
    load_path=$PTM_DIR

# finetune on NLVR2
NLVR2_DATA_DIR=/data/share/ViLT/data/NLVR2

# python run.py with \
#     data_root=$NLVR2_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=$NUM_NODES \
#     task_finetune_nlvr2_randaug \
#     per_gpu_batchsize=32 \
#     load_path=$PTM_DIR

# finetune on F30K IR/TR
F30K_DATA_DIR=/data/share/UNITER/origin_imgs/flickr30k

# python run.py with \
#     data_root=$F30K_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=$NUM_NODES \
#     task_finetune_irtr_f30k_randaug \
#     per_gpu_batchsize=4 \
#     load_path=$PTM_DIR

# finetune on COCO IR/TR
COCO_DATA_DIR=/data/share/UNITER/origin_imgs/coco_original

# python run.py with \
#     data_root=$COCO_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=$NUM_NODES \
#     task_finetune_irtr_coco_randaug \
#     per_gpu_batchsize=4 \
#     load_path=$PTM_DIR