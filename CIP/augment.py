import os, random
import pickle
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image
from torchvision.transforms.functional import scale
import numpy as np
import multiprocessing

from tqdm import tqdm
from joblib import Parallel, delayed

from coco_utils import (
    bbox2xy, loadImage, loadCaptions,
    image_vcat, image_hcat
)

from inst_desc_match import load_idmanns
from collections import defaultdict
from itertools import combinations

def inst_match_score(src_inst, tgt_inst, config):
    """
    """
    src_mask, tgt_mask = src_inst['mask'], tgt_inst['mask']

    src_size = (src_mask.shape[1], src_mask.shape[0])
    tgt_size = (tgt_mask.shape[1], tgt_mask.shape[0])
    scale_w, scale_h = src_size[0] / tgt_size[0], src_size[1] / tgt_size[1]

    # too small
    hw_scale = config.get('th_hw_scale', 0.7)
    if scale_w < hw_scale or scale_h < hw_scale:
        return 0.0
    
    # aspect ratio gap is too large
    aspect_ratio_src = src_size[0] / src_size[1]
    aspect_ratio_tgt = tgt_size[0] / tgt_size[1]

    if abs(aspect_ratio_src-aspect_ratio_tgt) > config.get('th_aspect_ratio_gap', 0.2):
        return 0.0

    # shape similarity (src_size => tgt_size)
    src_mask = Image.fromarray(src_mask)
    src_mask = np.array(src_mask.resize(tgt_size))
    sim = (src_mask * tgt_mask).sum() / ((src_mask+tgt_mask) != 0).sum()

    # VBG comparing
    return sim

def augment_captions(template_captions, descs, size=5):
    sample_descs = [descs[idx] for idx in np.random.randint(0, len(descs), size=size)]
    sample_template_captions = [template_captions[idx] for idx in np.random.randint(0, len(template_captions), size=size)]
    return [template_caption.replace('[OBJ]', desc) for desc, template_caption in zip(sample_descs, sample_template_captions) if '[OBJ]' in template_caption]

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

def get_inst_scores(idm_dir, instances, cat2ids):
    filename = f'{idm_dir}/inst_scores.pickle'
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            inst_scores = pickle.load(f)
    else:
        inst_scores = defaultdict(dict)
        for cat, ids in cat2ids.items():
            for i in tqdm(combinations(ids, 2), desc=f'processing score of {cat}'):
                tgt_inst = instances[i[0]]
                src_inst = instances[i[1]]
                match_score = inst_match_score(src_inst, tgt_inst)
                key = frozenset({tgt_inst['id'], src_inst['id']})
                inst_scores[cat][key] = match_score
        with open(filename, 'wb') as f:
            pickle.dump(inst_scores, f)
    return inst_scores

def get_inst_cat(inst, output_dir):
    if 'f30k' in output_dir or 'refcoco' in output_dir:
        cat = inst['attr']['category']
    else:
        cat = inst['attr']['category']['name']
    return cat

def test_fun(i, config):
    print(f'test function {i}')
    return i

def run_augment(config):
    print('[Run]: augmentation')
    random.seed(config['seed'])
    idm_dir = config['idm_dir']
    output_dir = config['output_dir']
    num_processes = config['num_processes']

    print('loading templates and instances')
    templates, instances, idm_config = load_idmanns(idm_dir)

    cat2ids = defaultdict(list)
    for iid, ins in instances.items():
        cat = get_inst_cat(ins, output_dir)
        cat2ids[cat].append(iid)

    os.makedirs(output_dir, exist_ok=True)

    def augmentation(i, keys):
        k = config['k']
        num_sample = config['num_sample']
        inpaint_dir = config['inpaint_dir']
        output_dir = config['output_dir']
        th_matchscore = config['th_matchscore']

        cnt = 0
        fout = open(f'{output_dir}/captions_{i}.txt', 'w')
        for template_key in tqdm(keys, desc=f'Augment Templates {i}'):
            t_cnt = 0
            template = templates[template_key]
            if template_key not in instances:
                continue
            tgt_inst = instances[template_key]
            cat = get_inst_cat(tgt_inst, output_dir)

            same_cat_instances = random.sample(cat2ids[cat], min(num_sample, len(cat2ids[cat])))
            for instance_key in same_cat_instances:
                if t_cnt >= k:
                    break
                if instance_key == template_key:
                    continue
                src_inst = instances[instance_key]
                # key = frozenset({tgt_inst['id'], src_inst['id']})
                # match_score = inst_scores[cat][key]
                match_score = inst_match_score(src_inst, tgt_inst, config)
                if match_score > th_matchscore:
                    aug_image, aug_captions = augment_template_with_instance(template, src_inst, f"{inpaint_dir}/{template['imgid']}-{template['inst_id']}_mask.png")
                    aug_image.save(f'{output_dir}/{template_key}-{instance_key}.jpg')
                    fout.write(f'{output_dir}/{template_key}-{instance_key}.jpg:\t{aug_captions}\n')
                    cnt += 1
                    t_cnt += 1
            if cnt % 1000 == 0:
                print(f'cnt: {cnt}')
        fout.close()
        return cnt
    
    print('augmentation')
    if num_processes == 1:
        cnt = augmentation(0, list(templates.keys()))
    else:
        split_keys = np.array_split(list(templates.keys()), num_processes)
        cnts = Parallel(n_jobs=num_processes)(delayed(augmentation)(i, split_keys[i]) for i in range(num_processes))
        # test = Parallel(n_jobs=num_processes)(delayed(test_fun)(i, config) for i in range(num_processes))
        cnt = sum(cnts)

    with open(f'{output_dir}/captions.txt', 'w') as fout:
        for i in range(num_processes):
            with open(f'{output_dir}/captions_{i}.txt', 'r') as fin:
                for line in fin:
                    fout.write(line)

    print(f'total augmented pairs: {cnt}')
    # int = 0
    # for template_key in tqdm(templates, desc='Augment Templates'):
    #     t_cnt = 0
    #     template = templates[template_key]
    #     if template_key not in instances:
    #         continue
    #     tgt_inst = instances[template_key]
    #     if 'f30k' in output_dir:
    #         cat = tgt_inst['attr']['category']
    #     else:
    #         cat = tgt_inst['attr']['category']['name']
    #     same_cat_instances = cat2ids[cat]
    #     for instance_key in same_cat_instances:
    #         if t_cnt >= k:
    #             break
    #         if instance_key == template_key:
    #             continue
    #         src_inst = instances[instance_key]
    #         # key = frozenset({tgt_inst['id'], src_inst['id']})
    #         # match_score = inst_scores[cat][key]
    #         match_score = inst_match_score(src_inst, tgt_inst, config)
    #         if match_score > th_matchscore:
    #             aug_image, aug_captions = augment_template_with_instance(template, src_inst, f"{inpaint_dir}/{template['imgid']}-{template['inst_id']}_mask.png")
    #             aug_image.save(f'{output_dir}/{template_key}-{instance_key}.jpg')
    #             fout.write(f'{output_dir}/{template_key}-{instance_key}.jpg:\t{aug_captions}\n')
    #             int += 1
    #             t_cnt += 1

if __name__ == '__main__':
    pass