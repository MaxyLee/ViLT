from collections import defaultdict
import re
import enum
import json
import torch
import pickle
import os, random, copy, time
import argparse
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image
import numpy as np
import nltk
import stanza
from gensim import models

from tqdm import tqdm
from omegaconf import OmegaConf

from coco_utils import (
    bbox2xy, loadImage, loadCaptions,
    image_vcat, image_hcat, draw_instance_bbox, draw_captions
)
from f30k_utils import get_sentence_data, get_annotations, get_segmentation

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
                'img_size': ann['img_size'],
                'object_template_captions': template_captions(captions_tokens, ann['inst_desc_map']), # ['A man standing next to [OBJ] on the ground .']
                'subject_template_captions': None # ['[SUB] on the ground .']
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
                    'category': ann['category'], # {supercategory, ..}
                    'vbg': None
                }
            }
            all_instances.append(instance)
        return all_instances

    @classmethod
    def from_f30k_anns(cls, image_path, imgid, captions, anns, segm):
        all_templates = cls.get_f30k_templates(imgid, captions, anns)
        all_instances = cls.get_f30k_instances(image_path, imgid, captions, anns, segm)

        return cls(image_path, imgid, all_templates, all_instances)

    @classmethod
    def get_f30k_templates(cls, imgid, captions, anns):
        all_templates = []
        img_size = (anns['height'], anns['width'])
        for phrase_id, bbox in anns['boxes'].items():
            template = {
                'imgid': imgid,
                'inst_id': phrase_id,
                'bbox': bbox[0], # use first bbox
                'img_size': img_size,
                'object_template_captions': f30k_template_captions(captions, phrase_id)
            }
            all_templates.append(template)
        return all_templates

    @classmethod
    def get_f30k_instances(cls, image_path, imgid, captions, anns, segm):
        all_instances = []
        image = Image.open(f'{image_path}/{imgid}.jpg')
        for phrase_id, bbox in anns['boxes'].items():
            x1, y1, x2, y2 = bbox[0]
            region = image.crop(bbox[0])
            try:
                mask = segm[phrase_id]['masks'][0, y1:y2, x1:x2].type(torch.uint8).numpy()
            except:
                # print('whoops')
                continue
            all_descs = [phrase['phrase'] for cap in captions for phrase in cap['phrases'] if phrase['phrase_id'] == phrase_id]
            for cap in captions:
                for phrase in cap['phrases']:
                    if phrase['phrase_id'] == phrase_id:
                        phrase_type = phrase['phrase_type'][0]
            instance = {
                'id': phrase_id,
                'region': np.array(region),
                'mask': mask,
                'object_descs': all_descs, # ['a dog', 'a small puppy']
                'subject_descs': None,     # ['a dog jumping', 'a small puppy playing with a frisbee']
                'attr': {
                    'category': phrase_type, # {supercategory, ..}
                    'vbg': None
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
    vector_sim = lambda x,y: (x*y).sum()/(np.linalg.norm(x)*np.linalg.norm(y))
    all_sims = [vector_sim(xx, y) for xx in x]
    return np.max(all_sims)

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
                if pos_tags[idx-1][1][:2] != 'JJ' and pos_tags[idx-1][1][0] != 'brown': # a brown and black dog
                    break
            elif pos_tags[idx][1][:2] == 'RB':
                if pos_tags[idx+1][1][:2] != 'JJ':
                    break
            elif pos_tags[idx][1][:2] not in ['NN', 'JJ', 'DT', 'CD'] and pos_tags[idx][1] != 'PRP$':
                break
            idx -= 1
        res[noun_idx] = (idx+1, noun_idx)

    return res

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return (ind,ind+sll-1)

def match_np(tree_str):
    # thank you, automata
    state = 's0'
    stack = []
    match = []
    tmp = ''
    for i, c in enumerate(tree_str):
        if state == 's0':
            if c == '(':
                if tree_str[i+1:i+3] == 'NP':
                    state = 's1'
                    tmp = '('
                    stack = ['(']
        elif state == 's1':
            tmp += c
            if c == '(':
                if tree_str[i+1:i+3] == 'NP':
                    stack = ['(']
                    tmp = '('
                else:
                    stack.append('(')
            elif c == ')':
                stack.pop()
                if stack == []:
                    state = 's0'
                    match.append(tmp)
    return match


def noun2np(tree):
    # return tokens because nltk tokenizer gets different tokens
    # hack for NNP matching issue
    tree_str = str(tree).replace('NNP', 'NN')
    tokens = [r[1] for r in re.findall(r'( \(.+? ([^\(]*?)\))', tree_str)]
    res = {}
    try:
        # nps = re.findall(r'(\(NP (\(((?!NP).)*? (\(*.*?\))*?\))+?\)?)', tree_str)
        nps = match_np(tree_str)
        for np in nps:
            # m = re.match(r'\(NP .*?\(NN\w? (.*?)\)\)', np[0])
            nouns = [r[1] for r in re.findall(r'( \(NN\w* (.*?)\))', np)]
            toks = [r[1] for r in re.findall(r'( \(.+? ([^\(]*?)\))', np)]

            indices = find_sub_list(toks, tokens)
            n_indices = [toks.index(n)+indices[0] for n in nouns]
            
            for idx in n_indices:
                res[idx] = indices
    except:
        import ipdb; ipdb.set_trace()
    return res, tokens

def location_parse(pos_tags):
    pass

def inst_desc_match_f30k(config, output_dir=None):
    print('[Run]: instance-description matching on f30k')
    root = config['root']
    img_path = f'{root}/flickr30k-images'
    ann_path = f'{root}/Annotations'
    cap_path = f'{root}/Sentences'
    seg_path = f'{root}/Segmentations'
    th_coverage = config.get('th_coverage', [0.1, 0.7])

    with open(f'{root}/karpathy/dataset_flickr30k.json', 'r') as f:
        dataset = json.load(f)
    images = dataset['images']

    imgids = [img['filename'][:-4] for img in images if img['split'] == 'train']

    idmAnns = []
    for imgid in tqdm(imgids):
        anns = get_annotations(f'{ann_path}/{imgid}.xml')
        caps = get_sentence_data(f'{cap_path}/{imgid}.txt')
        segm = get_segmentation(f'{seg_path}/{imgid}')

        img_area = anns['height'] * anns['width']

        filtered_boxes = defaultdict(list)
        for phrase_id, boxes in anns['boxes'].items():
            for box in boxes:
                box_area = (box[3] - box[1]) * (box[2] - box[0])
                coverage = float(box_area) / img_area
                if coverage > th_coverage[0] and coverage < th_coverage[1]:
                    filtered_boxes[phrase_id].append(box)
        anns['boxes'] = filtered_boxes

        idmAnn = IDMAnn.from_f30k_anns(img_path, imgid, caps, anns, segm)
        idmAnns.append(idmAnn)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_idmanns(idmAnns, config, output_dir=output_dir)
    elif 'output_dir' in config:
        output_dir = config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        save_idmanns(idmAnns, config, output_dir=output_dir)

    if config.get('visualize', False):
        os.makedirs(f'{output_dir}/result_imgs', exist_ok=True)
        idxs = random.sample(range(len(idmAnns)), 1000)
        for idx in tqdm(idxs, desc='Draw Templates'):
            idmAnn = idmAnns[idx]
            template = idmAnn.draw_templates_image()
            instance = idmAnn.draw_instances_image()
            image = image_vcat([template, instance])
            image.save(f'{output_dir}/result_imgs/{idmAnn.imgid}.png')

    return idmAnns

def run_inst_desc_match(config, output_dir=None):
    print('[Run]: instance-description matching')
    print('loading word2vec model into memory...')
    t = time.time()
    word2vec = models.KeyedVectors.load_word2vec_format(config['word2vec'], binary=True)
    print(f"Done (t={time.time()-t:.2f}s)")

    image_path = config['image_path']
    th_coverage = config.get('th_coverage', [0.1, 0.7])
    caption_anns = COCO(config['caption_annotation'])
    instance_anns = COCO(config['instance_annotation'])

    categories = config.get('categories', [])
    supercategories = config.get('supercategories', [])
    if len(categories) or len(supercategories):
        print(f'Specified Supercategories: {supercategories}')
        print(f'Specified Categories: {categories}')
        catIds = []
        if categories:
            catIds += instance_anns.getCatIds(catNms=categories)
        if supercategories:
            catIds += instance_anns.getCatIds(supNms=supercategories)
        imgIds = set()
        for catId in catIds:
            imgIds.update(instance_anns.getImgIds(catIds=[catId]))
        imgIds = list(imgIds)
    else:
        catIds = instance_anns.getCatIds()
        imgIds = instance_anns.getImgIds()

    print(f"augment images / total images: {len(imgIds)}/{len(instance_anns.getImgIds())}")

    with open(config['tree_path']) as f:
        id2tree = json.load(f)

    idmAnns = []
    for imgid in tqdm(imgIds, desc='Parse Templates'):
        # object-description alignment for a single image
        captions = loadCaptions(caption_anns, imgid)

        instance_annIds = instance_anns.getAnnIds(imgIds=[imgid])
        anns = instance_anns.loadAnns(instance_annIds)

        img_size = (instance_anns.imgs[imgid]['height'], instance_anns.imgs[imgid]['width'])
        img_area = img_size[0] * img_size[1]

        # filter out unsuitable instances
        filtered_anns = []
        for ann in anns:
            ann_catid = ann['category_id']
            if ann_catid not in catIds:
                continue
            inst_area = ann['bbox'][-2] * ann['bbox'][-1]
            coverage = float(inst_area) / img_area
            if coverage < th_coverage[0] or coverage > th_coverage[1]:
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
            # word2phrase_mapping = noun_word2phrase(tags, [i for i, _ in caption_noun_vecs])

            tree = id2tree[str(imgid)][caption]
            word2phrase_mapping_tree, tokens = noun2np(tree)

            # match ann_catname with caption_nouns
            for ann in anns:
                ann_id = ann['id']
                ann_catid = ann['category_id']
                ann_catname = instance_anns.cats[ann_catid]['name']
                ann_caption_mapping = []

                ann_vec = [word2vec[word] for word in ann_catname.split()]
                for idx, noun_vec in caption_noun_vecs:
                    sim = cosine_similarity(ann_vec, noun_vec)
                    
                    if sim > th:
                        if idx not in word2phrase_mapping_tree:
                            # print('whoops! word not in mapping...')
                            # print(tree)
                            # print(tags)
                            continue
                        l, r = word2phrase_mapping_tree[idx]
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

        if len(mapped_anns) == 0:
            continue

        for ann in mapped_anns:
            ann['category'] = instance_anns.cats[ann['category_id']]
            ann['mask'] = instance_anns.annToMask(ann)
            ann['img_size'] = (instance_anns.imgs[imgid]['height'], instance_anns.imgs[imgid]['width'])

        idmAnn = IDMAnn.from_mapped_anns(image_path, imgid, caption_tokens, mapped_anns)
        if len(idmAnn.templates) == 0 or idmAnn.instances == 0:
            import ipdb; ipdb.set_trace()
        idmAnns.append(idmAnn)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_idmanns(idmAnns, config, output_dir=output_dir)
    elif 'output_dir' in config:
        output_dir = config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        save_idmanns(idmAnns, config, output_dir=output_dir)

    if config.get('visualize', False):
        os.makedirs(f'{output_dir}/result_imgs', exist_ok=True)
        idxs = random.sample(range(len(idmAnns)), 1000)
        for idx in tqdm(idxs, desc='Draw Templates'):
            idmAnn = idmAnns[idx]
            template = idmAnn.draw_templates_image()
            instance = idmAnn.draw_instances_image()
            image = image_vcat([template, instance])
            image.save(f'{output_dir}/result_imgs/{idmAnn.imgid}.png')
    return idmAnns

############# Visualization Functions #############

def f30k_template_captions(captions, phrase_id):
    annoted_captions = []
    for cap in captions:
        sentence = cap['sentence']
        for phrase in cap['phrases']:
            if phrase['phrase_id'] == phrase_id:
                sentence = sentence[:phrase['first_word_index']] + '[OBJ]' + sentence[phrase['first_word_index']+len(phrase['phrase']):]
                break
        annoted_captions.append(sentence)
    return annoted_captions

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
        if token == '_' and inside_flag:
            annoted_caption.append('[OBJ]')
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

    print(f'Number of templates / total images: {len(templates)}/{len(idmAnns)}')

    json.dump({'image_path': config['image_path'], 'templates': templates}, open(f'{output_dir}/templates.json', 'w'))
    pickle.dump(instances, open(f'{output_dir}/instances.pickle', 'wb'))

def load_idmanns(output_dir, return_anns=False, load_instances=True):
    config = OmegaConf.load(f'{output_dir}/config.yaml')
    templates = json.load(open(f'{output_dir}/templates.json'))['templates']
    if load_instances:
        instances = pickle.load(open(f'{output_dir}/instances.pickle', 'rb'))
    else:
        instances = None

    if return_anns:
        image_path = config['image_path']
        image_anns = []
        keys = sorted(list(templates.keys()))

        tmp_key = keys[0]
        tmp_imgid = int(tmp_key.split('-')[0])
        tmp_templates = [templates[tmp_key]]
        tmp_instances = [instances[tmp_key]]
        for key in keys[1:]:
            imgid = int(key.split('-')[0])
            if imgid != tmp_imgid:
                image_anns.append(IDMAnn(image_path, tmp_imgid, tmp_templates, tmp_instances))
                tmp_imgid = imgid
                tmp_templates = []
                tmp_instances = []
            tmp_templates.append(templates[key])
            if load_instances:
                tmp_instances.append(instances[key])
        image_anns.append(IDMAnn(image_path, tmp_imgid, tmp_templates, tmp_instances))
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
    inst_desc_match_f30k()