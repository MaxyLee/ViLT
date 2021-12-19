import os
import json
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from data_process import run_preprocess, run_analysis
from extract_tree import run_extract_tree, run_extract_tree_refcoco
from inst_desc_match import run_inst_desc_match, inst_desc_match_f30k, inst_desc_match_refcoco
from phrase_grounding import run_phrase_grounding, run_segmentation
from crop_inpaint import run_crop_inpaint
from augment import run_augment
from evaluate import run_evaluate
from filtering import run_filter

if __name__ == '__main__':
    config = OmegaConf.load('configs/config.yaml')

    os.makedirs(config['tmp_path'], exist_ok=True)
    OmegaConf.save(config, f"{config['tmp_path']}/config.yaml")
    # run_preprocess(config['preprocess'])
    # run_analysis(config['analysis'])

    # run_segmentation(config['segmentation'])

    # run_extract_tree_refcoco(config['extract_tree_refcoco'])
    # run_extract_tree(config['extract_tree'])

    # run_inst_desc_match(config['inst_desc_match'])
    # inst_desc_match_f30k(config['inst_desc_match_f30k'])
    # inst_desc_match_refcoco(config['inst_desc_match_refcoco'])

    # run_phrase_grounding(config['phrase_grounding'])

    # run_crop_inpaint(config['crop_inpaint'])
    
    # run_augment(config['augment'])
    # run_evaluate(config['evaluate'])
    run_filter(config['filter'])