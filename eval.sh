set -ex

NUM_GPUS=1
NUM_NODES=1
PTM_DIR=/data/share/ViLT/pretrained/vilt_200k_mlm_itm.ckpt

# evaluate VQAv2
VQAv2_DATA_DIR=/data/share/UNITER/origin_imgs/coco_original
VQA_CKPT=/data/private/mxy/code/ViLT/result/finetune_vqa_randaug_seed0_from_vilt_200k_mlm_itm/version_1/checkpoints/epoch=9-step=24482.ckpt

# python run.py with \
#     data_root=$VQAv2_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=1 \
#     per_gpu_batchsize=64 \
#     task_finetune_vqa_randaug \
#     test_only=True \
#     load_path=$VQA_CKPT

# finetune on SNLI-VE
VE_DATA_DIR=/data/share/UNITER/origin_imgs/flickr30k
VE_CKPT=/data/private/mxy/code/ViLT/result/finetune_ve_seed0_from_vilt_200k_mlm_itm/version_17/checkpoints/epoch=3-step=7673.ckpt

CUDA_VISIBLE_DEVICES=4 python run.py with \
    data_root=$VE_DATA_DIR \
    num_gpus=$NUM_GPUS \
    num_nodes=$NUM_NODES \
    task_finetune_ve \
    per_gpu_batchsize=64 \
    test_only=True \
    load_path=$VE_CKPT

# evaluate NLVR2
NLVR2_DATA_DIR=/data/share/ViLT/data/NLVR2
NLVR2_CKPT=/data/private/mxy/code/ViLT/result/finetune_nlvr2_randaug_seed0_from_vilt_200k_mlm_itm/version_0/checkpoints/epoch=7-step=5399.ckpt

# python run.py with \
#     data_root=$NLVR2_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=1 \
#     per_gpu_batchsize=64 \
#     task_finetune_nlvr2_randaug \
#     test_only=True \
#     load_path=$NLVR2_CKPT

# evaluate F30K IR/TR
F30K_DATA_DIR=/data/share/UNITER/origin_imgs/flickr30k
F30K_IRTR_CKPT=/data/private/mxy/code/ViLT/result/finetune_irtr_f30k_randaug_seed0_from_vilt_200k_mlm_itm/version_0/checkpoints/epoch=9-step=5869.ckpt

# python run.py with \
#     data_root=$F30K_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=1 \
#     per_gpu_batchsize=4 \
#     task_finetune_irtr_f30k_randaug \
#     test_only=True \
#     load_path=$F30K_IRTR_CKPT

# evaluate COCO IR/TR
COCO_DATA_DIR=/data/share/UNITER/origin_imgs/coco_original
COCO_IRTR_CKPT=

# python run.py with \
#     data_root=$COCO_DATA_DIR \
#     num_gpus=$NUM_GPUS \
#     num_nodes=1 \
#     per_gpu_batchsize=4 \
#     task_finetune_irtr_coco_randaug \
#     test_only=True \
#     load_path=$COCO_IRTR_CKPT
