from omegaconf import OmegaConf
from inst_desc_match import run_inst_desc_match
from crop_inpaint import run_crop_inpaint
from augment import run_augment
from evaluate import run_evaluate

def quality_estimate(image_paths):
    pass

if __name__ == '__main__':
    config = OmegaConf.load('configs/config.yaml')
    # run_inst_desc_match(config['inst_desc_match'])
    # run_crop_inpaint(config['crop_inpaint'])
    # run_augment(config['augment'])
    run_evaluate(config['evaluate'])
