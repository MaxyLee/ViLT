import os
import json
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from data_process import run_preprocess, run_analysis
from inst_desc_match import run_inst_desc_match
from phrase_grounding import run_phrase_grounding
from crop_inpaint import run_crop_inpaint
from augment import run_augment
from evaluate import run_evaluate

if __name__ == '__main__':
    config = OmegaConf.load('configs/config.yaml')

    os.makedirs(config['tmp_path'], exist_ok=True)
    OmegaConf.save(config, f"{config['tmp_path']}/config.yaml")
    # run_preprocess(config['preprocess'])
    # run_analysis(config['analysis'])

    # run_inst_desc_match(config['inst_desc_match']) # TODO: all category
    run_phrase_grounding(config['phrase_grounding'])
    # run_crop_inpaint(config['crop_inpaint'])
    # run_augment(config['augment']) # TODO: all category, 类内比较，每张图扩展上限
    # run_evaluate(config['evaluate'])
