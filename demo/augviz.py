import argparse, os, sys, re, glob, math, time, json, random, functools

from numpy.lib.function_base import iterable
from pytorch_lightning.core import datamodule
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import copy
import io
import time
import torch
import numpy as np
import pandas as pd
import omegaconf
from omegaconf import OmegaConf
import streamlit as st
from streamlit import caching
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import T, default_collate

from vilt.modules.vilt_module import ViLTransformerSS

import hydra

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="demo/augviz_config.yaml",
        help="config files storing all_datas",
    )
    return parser.parse_args()

def tensor2pil(im_tensor):
    image = im_tensor.detach().cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image.clamp(0,1))
    return image

def aug2transform(aug):
    transform_keys = ['pixelbert']
    if aug == 'img':
        transform_keys = ['pixelbert_randaug']
    return transform_keys

############ Task Class Wrapper ############

class VQATask:

    def __init__(self, datamodule_config):
        self.datamodule_config = datamodule_config
        self.datamodule = hydra.utils.instantiate(datamodule_config)
        self.datamodule.setup('fit')

        self.answer2id = self.datamodule.answer2id
        self.num_class = self.datamodule.num_class
        self.id2answer = self.datamodule.id2answer

        self.datasets = {}

        self.collate = functools.partial(
            self.datamodule.train_dataset.collate, mlm_collator=self.datamodule.mlm_collator,
        )

    def load_dataset(self, aug_key='none', split='train'):
        transform_keys = aug2transform(aug_key)
        dataset = self.datamodule.dataset_cls(
            self.datamodule.data_dir,
            transform_keys,
            split=split,
            image_size=self.datamodule.image_size,
            max_text_len=self.datamodule.max_text_len,
            draw_false_image=self.datamodule.draw_false_image,
            draw_false_text=self.datamodule.draw_false_text,
            image_only=self.datamodule.image_only,
        )
        dataset.tokenizer = self.datamodule.tokenizer
        return dataset

    def display_example(self, data):
        image = data['image'][0]*0.5+0.5
        text = data['text'][0]
        st.image(tensor2pil(image))
        st.caption(text)
        st.json({
            'answer': data['vqa_answer'],
            'label': data['vqa_labels'],
            'score': data['vqa_scores'],
        })

    def test_example(self, pl_module, data):
        batch = self.collate([data])
        infer = pl_module.infer(batch, mask_text=False, mask_image=False)
        vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
        vqa_preds = vqa_logits.argmax(dim=-1)
        vqa_scores = [vqa_logits[i, pred].item() for i, pred in enumerate(vqa_preds)]
        vqa_preds = [self.id2answer[pred.item()] for pred in vqa_preds]
        outputs = {
            'preds': vqa_preds,
            'scores': vqa_scores
        }
        return outputs

        # id2answer = (
        #     pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        #     if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        #     else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
        # )
        # vqa_logits = output["vqa_logits"]
        # vqa_preds = vqa_logits.argmax(dim=-1)
        # vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
        # questions = batch["text"]
        # qids = batch["qid"]


    def get_length(self, split='train'):
        return len(getattr(self.datamodule, f'{split}_dataset', []))

    def name(self):
        return 'VQA'

task_classes = {
    'VQA': VQATask
}

hash_funcs = {
    VQATask: lambda x: x.name,
    ViLTransformerSS: lambda x: x.name
}

############ cached funcs ############

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_task(all_configs, task_name):
    return task_classes[task_name](all_configs[task_name]['datamodule'])

@st.cache(allow_output_mutation=True, suppress_st_warning=True, hash_funcs=hash_funcs)
def load_dataset(task, aug_key='none', split='train'):
    dataset = task.load_dataset(aug_key=aug_key, split=split)
    return dataset

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_configs(config_path):
    configs = OmegaConf.load(config_path)
    return configs

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_models(all_configs, task_name, device='cpu'):
    all_models = {}
    for k, ckpt_path in all_configs[task_name]['models'].items():
        all_models[k] = ViLTransformerSS.load_from_checkpoint(ckpt_path, map_location=device)
        all_models[k].name = k
    return all_models

############ ui funcs ############

def select_task(tasks):
    task = st.sidebar.selectbox('Select a task', options=tasks)
    return task

def select_dataset(split_options=['train', 'val'], aug_options=['none', 'img', 'txt', 'img+txt']):
    split = st.sidebar.selectbox('Select a data split', options=split_options)
    augs = st.sidebar.multiselect(label='Selected augmentation strategy', options=aug_options, default=aug_options[0])
    return split, augs

def select_index(max_index=0):
    index_placeholder = st.sidebar.empty()
    random_select_btn = st.sidebar.button('Sample', key='random_select')
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = 0
    if st.session_state.get("random_select", False):
        st.session_state.selected_index = random.randint(0, max_index)
    index = index_placeholder.number_input(f"Example Index (Size: {max_index})", value=st.session_state['selected_index'], min_value=0, max_value=max_index)
    return index


if __name__ == '__main__':
    args = parse_args()
    all_configs = load_configs(args.config)

    # choose task
    task_name = select_task(list(all_configs.keys()))
    split, augs = select_dataset()
    selected_task = load_task(all_configs, task_name)

    # load all datasets
    all_datasets = {}
    for aug in augs:
        all_datasets[f'{split}_{aug}'] = load_dataset(selected_task, aug_key=aug, split=split)
    
    # load all models
    all_models = load_models(all_configs, task_name)

    index = select_index(selected_task.get_length(split))

    cols = st.columns(len(all_datasets))
    for col, (k, v) in zip(cols, all_datasets.items()):
        with col:
            st.caption(k)
            example = v[index]
            selected_task.display_example(example)
            for module_k, pl_module in all_models.items():
                st.json(selected_task.test_example(pl_module, example))
    
