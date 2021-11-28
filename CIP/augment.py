import os, random
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image
from torchvision.transforms.functional import scale
import numpy as np

from tqdm import tqdm

from coco_utils import (
    bbox2xy, loadImage, loadCaptions,
    image_vcat, image_hcat
)

from inst_desc_match import load_idmanns

def inst_match_score(src_inst, tgt_inst):
    """ shape similarity (src_size => tgt_size)
    """
    src_mask, tgt_mask = src_inst['mask'], tgt_inst['mask']

    src_size = (src_mask.shape[1], src_mask.shape[0])
    tgt_size = (tgt_mask.shape[1], tgt_mask.shape[0])
    scale_w, scale_h = src_size[0] / tgt_size[0], src_size[1] / tgt_size[1]

    if scale_w < 0.5 or scale_h < 0.5:
        return 0.0
    
    # mask similarity
    src_mask = Image.fromarray(src_mask)
    src_mask = np.array(src_mask.resize(tgt_size))
    sim = (src_mask * tgt_mask).sum() / ((src_mask+tgt_mask) != 0).sum()

    # VBG comparing
    return sim

def augment_captions(template_captions, descs, size=5):
    sample_descs = [descs[idx] for idx in np.random.randint(0, len(descs), size=size)]
    sample_template_captions = [template_captions[idx] for idx in np.random.randint(0, len(template_captions), size=size)]
    return [template_caption.replace('[OBJ]', desc) for desc, template_caption in zip(sample_descs, sample_template_captions)]

def augment_template_with_instance(template, instance, image_filepath):
    image = Image.open(image_filepath)
    
    # paste instance
    tgt_bbox = bbox2xy(template['bbox'])
    tgt_size = (tgt_bbox[2]-tgt_bbox[0], tgt_bbox[3]-tgt_bbox[1]) # x2 - x1, y2 - y1

    src_region, src_mask = instance['region'], instance['mask'] # np.ndarray
    src_region, src_mask = Image.fromarray(src_region), Image.fromarray(src_mask*255)
    
    src_region = src_region.resize(tgt_size)
    src_mask = src_mask.resize(tgt_size)
    image.paste(src_region, box=tgt_bbox, mask=src_mask)

    # generate captions by filling in the blanks in each template caption
    captions = augment_captions(template['object_template_captions'], instance['object_descs'])
    return image, captions

def run_augment(config):
    print('[Run]: augmentation')
    idm_dir = config['idm_dir']
    inpaint_dir = config['inpaint_dir']
    output_dir = config['output_dir']
    th_matchscore = config['th_matchscore']
    templates, instances, idm_config = load_idmanns(idm_dir)

    os.makedirs(output_dir, exist_ok=True)
    fout = open(f'{output_dir}/captions.txt', 'w')

    int = 0
    for template_key in tqdm(templates, desc='Augment Templates'):
        template = templates[template_key]
        tgt_inst = instances[template_key]
        for instance_key in instances:
            if instance_key == template_key:
                continue
            src_inst = instances[instance_key]
            match_score = inst_match_score(src_inst, tgt_inst)
            if match_score > th_matchscore:
                aug_image, aug_captions = augment_template_with_instance(template, src_inst, f"{inpaint_dir}/{template['imgid']}-{template['inst_id']}_mask.png")
                aug_image.save(f'{output_dir}/{template_key}-{instance_key}.jpg')
                fout.write(f'{output_dir}/{template_key}-{instance_key}.jpg:\t{aug_captions}\n')
                int += 1

    print(f'total augmented pairs: {int}')

if __name__ == '__main__':
    pass