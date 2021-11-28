import os
import json
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from inst_desc_match import run_inst_desc_match
from crop_inpaint import run_crop_inpaint
from augment import run_augment
from evaluate import run_evaluate

def run_preprocess(config):
    print('[Run]: preprocess')
    image_ids = np.load(config['image_ids'])

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'val']:
        print(f'preprocess annotations[{split}]...')
        caption_annotation = json.load(open(config['caption_annotation'][split]))
        instance_annotation = json.load(open(config['instance_annotation'][split]))

        info = caption_annotation['info']
        licenses = caption_annotation['licenses']
        categories = instance_annotation['categories']

        caption_mapping = {}
        instance_mapping = {}
        images_mapping = {}
        for ann in tqdm(caption_annotation['annotations'], desc='Traverse Captions'):
            image_id = ann['image_id']
            id = ann['id']
            caption = ann['caption']
            if image_id not in caption_mapping:
                caption_mapping[image_id] = [(id, caption)]
            else:
                caption_mapping[image_id].append((id, caption))
        
        for ann in tqdm(instance_annotation['annotations'], desc='Traverse Instances'):
            image_id = ann.pop('image_id')
            id = ann.pop('id')
            if image_id not in instance_mapping:
                instance_mapping[image_id] = [(id, ann)]
            else:
                instance_mapping[image_id].append((id, ann))

        for image in tqdm(caption_annotation['images'], desc='Traverse Images'):
            images_mapping[image['id']] = image

        new_caption_anns = []
        new_instance_anns = []
        new_images = []
        for image_id in tqdm(image_ids, desc='Extract small dataset'):
            if image_id in caption_mapping:
                for id, caption in caption_mapping[image_id]:
                    new_caption_anns.append({'image_id': int(image_id), 'id': id, 'caption': caption})
            if image_id in instance_mapping:
                for id, inst_ann in instance_mapping[image_id]:
                    inst_ann.update({'image_id': int(image_id), 'id': id})
                    new_instance_anns.append(inst_ann)
            if image_id in images_mapping:
                new_images.append(images_mapping[image_id])

        json.dump({'info': info, 'licenses': licenses, 'images': new_images, 'annotations': new_caption_anns}, open(f'{output_dir}/captions_{split}.json', 'w'))
        json.dump({'info': info, 'licenses': licenses, 'images': new_images, 'annotations': new_instance_anns, 'categories': categories}, open(f'{output_dir}/instances_{split}.json', 'w'))


if __name__ == '__main__':
    config = OmegaConf.load('configs/config.yaml')
    # run_preprocess(config['preprocess'])

    run_inst_desc_match(config['inst_desc_match']) # TODO: all category
    run_crop_inpaint(config['crop_inpaint'])
    run_augment(config['augment']) # TODO: all category, 类内比较，每张图扩展上限
    run_evaluate(config['evaluate'])
