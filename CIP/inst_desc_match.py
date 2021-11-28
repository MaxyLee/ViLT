import enum
import json
import pickle
import os, random, copy, time
import argparse
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image
import numpy as np
import nltk
from gensim import models

from tqdm import tqdm
from omegaconf import OmegaConf

from coco_utils import (
    bbox2xy, loadImage, loadCaptions,
    image_vcat, image_hcat, draw_instance_bbox, draw_captions
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default='configs/idm.yaml',
        help="run instance-description matching with given configs",
    )
    return parser.parse_args()

############# Match Classes #############

colornames = ['red', 'lightgreen', 'SandyBrown', 'RoyalBlue']

class IDMAnn:
    """
    Instance-Description Matched Annotations for a single image
    """
    def __init__(self, image_path, imgid, templates, instances):
        self.image_path = image_path
        self.imgid = imgid
        self.templates = templates
        self.instances = instances

    @classmethod
    def from_mapped_anns(cls, image_path, imgid, captions_tokens, mapped_anns):
        all_templates = cls.get_templates(image_path, imgid, captions_tokens, mapped_anns)
        all_instances = cls.get_instances(image_path, imgid, mapped_anns)

        return cls(image_path, imgid, all_templates, all_instances)

    @classmethod
    def get_templates(cls, image_path, imgid, captions_tokens, mapped_anns):
        all_templates = []
        for ann in mapped_anns:
            template = {
                'imgid': imgid,
                'inst_id': ann['id'],
                'bbox': ann['bbox'],
                'object_template_captions': template_captions(captions_tokens, ann['inst_desc_map']), # ['A man standing', 'next', 'to', '[TPL]', 'on', 'the', 'ground', '.']
                'subject_template_captions': None
            }
            all_templates.append(template)
        return all_templates

    @classmethod
    def get_instances(cls, image_path, imgid, mapped_anns):
        image = loadImage(image_path, imgid)
        all_instances = []
        for ann in mapped_anns:
            x1, y1, x2, y2 = bbox2xy(ann['bbox'])
            region = image.crop((x1, y1, x2, y2))
            mask = ann['mask'][y1:y2, x1:x2]
            all_descs = [xx[1].lower() for x in ann['inst_desc_map'] for xx in x]
            instance = {
                'id': ann['id'],
                'region': np.array(region),
                'mask': mask,
                'object_descs': all_descs, # ['a dog', 'a small puppy']
                'subject_descs': None,     # ['a dog jumping', 'a small puppy playing with a frisbee']
                'attr': {
                    'category': ann['category'],
                }
            }
            all_instances.append(instance)
        return all_instances

    def draw_templates_image(self):
        image = loadImage(self.image_path, self.imgid)
        cnt = 0
        all_template_captions = []
        for template, color in zip(self.templates, colornames):
            label = f"inst-{template['inst_id']}"
            image = draw_instance_bbox(image, template['bbox'], color=color, label=label)
            cnt += 1
            all_template_captions.append((label, color, template['object_template_captions']))
        
        all_caption_images = []
        for label, color, template_captions in all_template_captions:
            caption_image = draw_captions(image.size[1], template_captions, title=label+':', color=color)
            all_caption_images.append(caption_image)

        template_image = image_hcat([image] + all_caption_images)
        return template_image

    def draw_instances_image(self, min_height=200, masked=True):
        image = loadImage(self.image_path, self.imgid)
        all_instance_images = [image]
        for instance in self.instances:
            region = Image.fromarray(instance['region'])
            mask = instance['mask']
            descs = instance['object_descs']
            if masked:
                img = Image.new('RGB', region.size, (255, 255, 255))
                img.paste(region, mask=Image.fromarray(mask*255))
            else:
                img = region
            desc_image = draw_captions(max(img.size[1], min_height), descs)
            instance_image = image_hcat([img, desc_image])
            all_instance_images.append(instance_image)
        return image_vcat(all_instance_images)

############# Match Functions #############

def cosine_similarity(x, y):
    return (x*y).sum()/(np.linalg.norm(x)*np.linalg.norm(y))

def instance_similarity_score(src_instance, tgt_instance):
    """ shape similarity (src_size => tgt_size)
    """
    src_mask, tgt_mask = src_instance['mask'], tgt_instance['mask']

    # mask similarity
    tgt_size = (tgt_mask.shape[1], tgt_mask.shape[0])
    src_mask = Image.fromarray(src_mask)
    src_mask = np.array(src_mask.resize(tgt_size))
    sim = (src_mask * tgt_mask).sum() / ((src_mask+tgt_mask) != 0).sum()
    return sim, src_mask

def noun_word2phrase(pos_tags, noun_ids):
    res = {}
    for noun_idx in noun_ids:
        idx = noun_idx - 1
        while idx >= 0:
            if pos_tags[idx][1][:2] == 'CC':
                if idx == 0:
                    break
                if pos_tags[idx-1][1][:2] != 'JJ':
                    break
            elif pos_tags[idx][1][:2] == 'RB':
                if pos_tags[idx+1][1][:2] != 'JJ':
                    break
            elif pos_tags[idx][1][:2] not in ['NN', 'JJ', 'DT', 'CD'] and pos_tags[idx][1] != 'PRP$':
                break
            idx -= 1
        res[noun_idx] = (idx+1, noun_idx)

    return res

def location_parse(pos_tags):
    pass

def run_inst_desc_match(config, output_dir=None):
    print('[Run]: instance-description matching')
    print('loading word2vec model into memory...')
    t = time.time()
    word2vec = models.KeyedVectors.load_word2vec_format(config['word2vec'], binary=True)
    print(f"Done (t={time.time()-t:.2f}s)")

    image_path = config['image_path']
    caption_anns = COCO(config['caption_annotation'])
    instance_anns = COCO(config['instance_annotation'])

    nms = config['categories']
    catIds = instance_anns.getCatIds(catNms=nms)
    imgIds = instance_anns.getImgIds(catIds=catIds)

    idmAnns = []
    for imgid in tqdm(imgIds, desc='Parse Templates'):
        # object-description alignment for a single image
        captions = loadCaptions(caption_anns, imgid)

        instance_annIds = instance_anns.getAnnIds(imgIds=[imgid])
        anns = instance_anns.loadAnns(instance_annIds)

        filtered_anns = []
        for ann in anns:
            ann_catid = ann['category_id']
            if ann_catid not in catIds:
                continue
            filtered_anns.append(ann)
        anns = filtered_anns   

        instance_descrip_mapping = {}
        caption_tokens = []
        mapped_anns = []

        th = config['th_wordsim']

        for caption in captions:
            tokens = nltk.word_tokenize(caption)
            caption_tokens.append(tokens)
            tags = nltk.pos_tag(tokens)

            # get all nouns from the caption
            caption_noun_vecs = []
            for i, (word, pos_tag) in enumerate(tags):
                if pos_tag[0] == 'N':
                    if word not in word2vec:
                        continue
                    caption_noun_vecs.append((i, word2vec[word]))

            # noun word to noun phrase
            word2phrase_mapping = noun_word2phrase(tags, [i for i, _ in caption_noun_vecs])

            # match ann_catname with caption_nouns
            for ann in anns:
                ann_id = ann['id']
                ann_catid = ann['category_id']
                ann_catname = instance_anns.cats[ann_catid]['name']
                ann_caption_mapping = []

                ann_vec = word2vec[ann_catname]
                for idx, noun_vec in caption_noun_vecs:
                    sim = cosine_similarity(ann_vec, noun_vec)
                    if sim > th:
                        l, r = word2phrase_mapping[idx]
                        ann_caption_mapping.append(((l, r+1), 
                                                    ' '.join(tokens[l:r+1])))

                if ann_id not in instance_descrip_mapping:
                    instance_descrip_mapping[ann_id] = []
                instance_descrip_mapping[ann_id].append(ann_caption_mapping)

        # filter out unimportant objects
        for ann in anns:
            ann_id = ann['id']
            if sum([len(x) for x in instance_descrip_mapping[ann_id]]) != 0:
                ann['inst_desc_map'] = instance_descrip_mapping[ann_id]
                mapped_anns.append(ann)

        # image = loadImage(inpaint_image_path, imgid, fmt='{imgid}_mask.png')

        for ann in mapped_anns:
            ann['category'] = instance_anns.cats[ann['category_id']]
            ann['mask'] = instance_anns.annToMask(ann)

        idmAnn = IDMAnn.from_mapped_anns(image_path, imgid, caption_tokens, mapped_anns)
        idmAnns.append(idmAnn)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_idmanns(idmAnns, config, output_dir=output_dir)
    elif 'output_dir' in config:
        output_dir = config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        save_idmanns(idmAnns, config, output_dir=output_dir)
    return idmAnns

############# Visualization Functions #############

def template_captions(caption_tokens, inst_desc_mapping):
    """
    Args:
        caption_tokens: [['a', 'dog', 'in', 'the', 'yard'], ['a', 'man', 'is', 'with', 'his', 'dog']]
        inst_desc_mapping: [[((0, 2), 'a dog')], [((4,6), 'his dog')]]
    """
    annoted_captions = []
    for caption, caption_inst_desc in zip(caption_tokens, inst_desc_mapping):
        caption = copy.copy(caption)
        for ((l, r), _) in caption_inst_desc:
            caption[l:r] = ['_' for _ in range(r-l)]
        annoted_caption = []
        inside_flag = False
        for token in caption:
            if token == '_' and not inside_flag:
                inside_flag = True
            elif token == '_' and inside_flag:
                continue
            elif token != '_' and inside_flag:
                inside_flag = False
                annoted_caption.append('[OBJ]')
                annoted_caption.append(token)
            else:
                annoted_caption.append(token)
        annoted_captions.append(' '.join(annoted_caption))
    return annoted_captions

def draw_instance(image, bbox, mask, min_height=200, masked=True):
    """draw instance image: image_hcat([instance(croped), descriptios])
    
    Args:
        masked (Bool): whether to mask out unrelated pixels in the bbox
    """
    x1, y1, x2, y2 = bbox2xy(bbox)
    region = image.crop((x1, y1, x2, y2))
    mask = mask[y1:y2, x1:x2]
    if not masked:
        return region
    else:
        paste_bbox = [0, 0, x2-x1, y2-y1]
        image_size = region.size
        if image_size[1] < min_height:
            margin_top = int((min_height - image_size[1])/2)
            paste_bbox = [0, margin_top, x2-x1, y2-y1+margin_top]
            image_size = (image_size[0], min_height)
        new_img = Image.new('RGB', image_size, (255, 255, 255))
        new_img.paste(region, box=paste_bbox, mask=Image.fromarray(mask*255))
        return new_img, mask

def save_idmanns(idmAnns, config, output_dir='idm_results'):
    print('saving results...')
    OmegaConf.save(config, f'{output_dir}/config.yaml')
    templates = {}
    instances = {}
    for idmAnn in idmAnns:
        imgid = idmAnn.imgid
        for idx, template in enumerate(idmAnn.templates):
            templates[f"{imgid}-{template['inst_id']}"] = template
        for idx, instance in enumerate(idmAnn.instances):
            instances[f"{imgid}-{instance['id']}"] = instance

    json.dump({'image_path': config['image_path'], 'templates': templates}, open(f'{output_dir}/templates.json', 'w'))
    pickle.dump(instances, open(f'{output_dir}/instances.pickle', 'wb'))

def load_idmanns(output_dir, return_anns=False):
    config = OmegaConf.load(f'{output_dir}/config.yaml')
    templates = json.load(open(f'{output_dir}/templates.json'))['templates']
    instances = pickle.load(open(f'{output_dir}/instances.pickle', 'rb'))

    if return_anns:
        image_path = config['image_path']
        image_anns = []
        keys = sorted(list(templates.keys()))

        tmp_key = keys[0]
        tmp_imgid = int(tmp_key.split('-')[0])
        tmp_templates = []
        tmp_instances = []
        for key in keys[1:]:
            imgid = int(key.split('-')[0])
            if imgid != tmp_imgid:
                image_anns.append(IDMAnn(image_path, tmp_imgid, tmp_templates, tmp_instances))
                tmp_imgid = imgid
                tmp_templates = []
                tmp_instances = []
            else:
                tmp_templates.append(templates[key])
                tmp_instances.append(instances[key])
        return image_anns
    else:
        return templates, instances, config

if __name__ == '__main__':
    # args = parse_args()
    # config = OmegaConf.load(args.config)
    # idmAnns = run_inst_desc_match(config)
    
    # output_dir = config['output_dir']
    # os.makedirs(output_dir, exist_ok=True)
    # save_idmanns(idmAnns, config, output_dir=output_dir)

    idmAnns = load_idmanns('tmp/idm_results')

    # os.makedirs('test/templates', exist_ok=True)
    # os.makedirs('test/instances', exist_ok=True)

    # for idmAnn in idmAnns:
    #     template = idmAnn.draw_templates_image()
    #     instance = idmAnn.draw_instances_image()
    #     template.save(f'test/templates/{idmAnn.imgid}.png')
    #     instance.save(f'test/instances/{idmAnn.imgid}.png')

    import ipdb; ipdb.set_trace()