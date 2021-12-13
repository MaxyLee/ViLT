import os
import time
import json
import shutil
import signal
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
    output_dir = config['template_path']

    print('saving_images')
    os.makedirs(output_dir, exist_ok=True)
    for label, template in tqdm(templates.items(), desc='Save images and masks'):
        img_name = f"{template['imgid']}.jpg" if 'f30k' in output_dir else f"{template['imgid']:0>12d}.jpg"
        shutil.copyfile(f"{image_path}/{img_name}", f'{output_dir}/{label}.png')
        # shutil.copyfile(f"{image_path}/{template['imgid']}.jpg", f'{output_dir}/{label}.png')
        img_size = template['img_size']

        mask = np.zeros(img_size, dtype=np.uint8)
        if 'f30k' in output_dir:
            x1, y1, x2, y2 = template['bbox']
        else:
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

    data_split = len(device)
    if data_split == 1:
        command =  f"{environ} python3 {src_file} model.path={inpaint_config['model_path']} indir={indir} outdir={outdir} device={device}"
        subprocess.run(command, shell=True)
    else:
        # split data
        process_pool = []
        label_splits = np.array_split(list(templates.keys()), data_split)
        for i, labels in enumerate(label_splits):
            indir_i = f"{indir}/subdir-{i}"
            device_i = device[i]
            os.makedirs(indir_i, exist_ok=True)
            for label in tqdm(labels):
                shutil.copyfile(f"{output_dir}/{label}.png", f"{indir_i}/{label}.png")
                shutil.copyfile(f"{output_dir}/{label}_mask.png", f"{indir_i}/{label}_mask.png")
            command_i =  f"{environ} exec python3 {src_file} model.path={inpaint_config['model_path']} indir={indir_i} outdir={outdir} device={device_i} &"
            process_pool.append(subprocess.Popen(command_i, shell=True))
        
        def terminate(signum, frame):
            print('KeyboardInterupt: kill all subprocesses...')
            for p in process_pool:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            exit()

        signal.signal(signal.SIGINT, terminate)
        time.sleep(1000)
        while True:
            status = [p.poll() for p in process_pool]
            if None in status:
                time.sleep(10)
            else:
                print('[Run] crop-inpaint: finished!')
                break


if __name__ == '__main__':
    args = parse_args()
    config = OmegaConf.load(args.config)
    run_crop_inpaint(config)