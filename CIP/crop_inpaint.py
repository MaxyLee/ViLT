import os
import json
import shutil
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from omegaconf import OmegaConf

from PIL import Image

from coco_utils import (
    loadImage, bbox2xy
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default='configs/crop_inpaint.yaml',
        help="run instance-description matching with given configs",
    )
    return parser.parse_args()

def run_crop_inpaint(config):
    """
    template = {
            'imgid': self.imgid,
            'inst_id': ann['id'],
            'object_template_captions': template_captions(self.captions_tokens, ann['inst_desc_map']), # ['A man standing', 'next', 'to', '[TPL]', 'on', 'the', 'ground', '.']
            'subject_template_captions': None
        }
    """
    print('[Run]: crop-inpaint')
    annotations = json.load(open(config['template_annotation']))
    image_path, templates = annotations['image_path'], annotations['templates']

    print('saving_images')
    output_dir = config['template_path']
    os.makedirs(output_dir, exist_ok=True)
    for label, template in tqdm(templates.items(), desc='Save images and masks'):
        shutil.copyfile(f"{image_path}/{template['imgid']:0>12d}.jpg", f'{output_dir}/{label}.png')
        img_size = template['img_size']

        mask = np.zeros(img_size, dtype=np.uint8)
        x1, y1, x2, y2 = bbox2xy(template['bbox'])
        mask[y1:y2, x1:x2] = 255
        mask_image = Image.fromarray(mask)
        mask_image.save(f'{output_dir}/{label}_mask.png')

    # run lama
    print('run lama inpainting')
    inpaint_config = config['inpaint']
    environ = f"MKL_THREADING_LAYER=GNU TORCH_HOME={inpaint_config['code_path']} && PYTHONPATH={inpaint_config['code_path']}"
    src_file = f"{inpaint_config['code_path']}/bin/predict.py"
    indir = os.path.abspath(inpaint_config['indir'])
    outdir = os.path.abspath(inpaint_config['outdir'])
    device = inpaint_config['device']
    command =  f"{environ} python3 {src_file} model.path={inpaint_config['model_path']} indir={indir} outdir={outdir} device={device}"
    subprocess.run(command, shell=True)

if __name__ == '__main__':
    args = parse_args()
    config = OmegaConf.load(args.config)
    run_crop_inpaint(config)