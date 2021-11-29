import os
import json
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO

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

def run_analysis(config):
    fout = open(f"{config['output_dir']}/summaries.txt", 'w')
    for split in ['train', 'val']:
        fout.write('='*15 + split + '='*15 + '\n\n')
        instance_anns = COCO(config['instance_annotation'][split])
        total_images = len(instance_anns.getImgIds())
        cat_summaries = []
        supercat_summaries = {}
        for k, cat in instance_anns.cats.items():
            imgIds = instance_anns.getImgIds(catIds=[cat['id']])
            cat['num_images'] = len(imgIds)
            cat_summaries.append(cat)
            supercat = cat['supercategory']
            if supercat not in supercat_summaries:
                supercat_summaries[supercat] = [[], 0]
            supercat_summaries[supercat][0].append(cat['name'])
            supercat_summaries[supercat][1] += cat['num_images']
        
        fout.write('[STATS] Categories:\n')
        cats = sorted(cat_summaries, key=lambda x: x.get('num_images', 0), reverse=True)
        for cat in cats:
            ratio = float(cat['num_images']) / total_images * 100
            fout.write(f"{cat['id']}\t{cat['name']}\t{cat['supercategory']}\t{cat['num_images']}\t{ratio:.2f}%\n")

        fout.write('\n[STATS] Supercategories:\n')
        supercats = sorted(supercat_summaries.items(), key=lambda x: x[1][1], reverse=True)
        for supercat in supercats:
            ratio = float(supercat[1][1]) / total_images * 100
            fout.write(f"{supercat[0]}: \t{supercat[1][1]}\t{ratio:.2f}%\t\n{supercat[1][0]}\n\n")

    fout.close()