import enum
import os, random, copy, time
import argparse
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import nltk
from gensim import models

from tqdm import tqdm
from omegaconf import OmegaConf

from coco_utils import (
    bbox2xy, loadImage, loadCaptions,
    image_vcat, image_hcat
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        required=True,
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="sample",
        help="path for samples output",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs='*',
        help="gpus",
    )
    return parser.parse_args()

############# Match Classes #############

class IDMAnn:
    """
    Instance-Description Matched Annotations for a single image
    """
    def __init__(self, image_path, imgid, captions_tokens, mapped_anns, inpaint_image_path=None):
        self.image_path = image_path
        self.inpaint_image_path = inpaint_image_path
        self.imgid = imgid
        self.captions_tokens = captions_tokens
        self.mapped_anns = mapped_anns

    def get_templates(self):
        all_templates = []
        for ann in self.mapped_anns:
            template = {
                'image_path': self.image_path,
                'imgid': self.imgid,
                'inst_id': ann['id'],
                'object_template_captions': template_captions(self.captions_tokens, ann['inst_desc_map']), # ['A man standing', 'next', 'to', '[TPL]', 'on', 'the', 'ground', '.']
                'subject_template_captions': None
            }
            all_templates.append(template)
        return all_templates

    def get_instances(self):
        image = loadImage(self.image_path, self.imgid)
        all_instances = []
        for ann in self.mapped_anns:
            x1, y1, x2, y2 = bbox2xy(ann['bbox'])
            region = image.crop((x1, y1, x2, y2))
            mask = ann['mask'][y1:y2, x1:x2]
            all_descs = [[xx[1] for x in ann['inst_desc_map'] for xx in x]]
            instance = {
                'id': ann['id'],
                'region': region,
                'mask': mask,
                'object_descs': all_descs, # ['a dog', 'a small puppy']
                'subject_descs': None,     # ['a dog jumping', 'a small puppy playing with a frisbee']
                'attr': {
                    'category': ann['category'],
                }
            }
            all_instances.append(instance)
        return all_instances

    def get_template_image(self, masked=False):
        if masked and self.inpaint_image_path is not None:
            image = loadImage(self.inpaint_image_path, self.imgid, fmt='{imgid}_mask.png')
        else:
            image = loadImage(self.image_path, self.imgid)
        template_image = draw_template(image, self.mapped_anns, self.captions_tokens, masked=masked)
        return template_image

    def get_instance_image(self, image=None):
        image = loadImage(self.image_path, self.imgid)
        all_instance_images = []
        for ann in self.mapped_anns:
            instance, _ = draw_instance(image, ann)
            descs = [cap_pairs[0][1] for cap_pairs in ann['inst_desc_map'] if len(cap_pairs)]
            desc_image = draw_captions(instance.size[1], descs)
            instance_image = image_hcat([instance, desc_image])
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

def inst_desc_match(config):
    print('run instance-description matching')
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

        idmAnn = IDMAnn(image_path, imgid, caption_tokens, mapped_anns)
        idmAnns.append(idmAnn)

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

def draw_instance_bbox(image, ann, color='red', label='bbox'):
    """show instance bbox in the image
    """
    bbox = ann['bbox']
    x1, y1, x2, y2 = bbox2xy(bbox)
    
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    label_size = draw.textsize(label, font)
    text_origin = np.array([x1, y1-label_size[1]])

    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    draw.rectangle([tuple(text_origin), tuple(text_origin+label_size)], fill=color)
    draw.text(text_origin, str(label), fill=(255,255,255), font=font)

    del draw
    return image

def draw_captions(height, captions, title=None, color='black'):
    """draw caption images with specified height and adaptive width
    """
    margin_left, margin_right = 10, 10
    margin_between = 10

    image = Image.new('RGB', (256, height), (255, 255, 255))

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

    textsizes = [draw.textsize(caption, font) for caption in captions]
    max_width = max([textsize[0] for textsize in textsizes])

    total_height = sum([textsize[1] for textsize in textsizes])
    total_height += len(textsizes) * margin_between
    margin_top = int((height - total_height) / 2)

    image = image.resize((max_width+margin_left+margin_right, height))
    draw = ImageDraw.Draw(image)

    text_origin = np.array([margin_left, margin_top])

    if title is not None:
        title_textsize = draw.textsize(title, font)
        draw.text(text_origin, str(title), fill=color, font=font)
        text_origin[1] += title_textsize[1] + margin_between

    for caption, textsize in zip(captions, textsizes):
        draw.text(text_origin, str(caption), fill=(0,0,0), font=font)
        text_origin[1] += textsize[1] + margin_between

    return image

colornames = ['red', 'lightgreen', 'SandyBrown', 'RoyalBlue']

def draw_template(image, mapped_anns, caption_tokens, masked=False):
    """draw template image: image_hcat([image(instance marked), captions])
    
    Args:
        masked (Bool): whether to mask out descriptions corresponding to instances
    """
    all_template_captions = {}
    for ann, color in zip(mapped_anns, colornames):
        ann_catname = f"{ann['category']['name']}-{ann['id']}"

        image = draw_instance_bbox(image, ann, color=color, label=ann_catname)

        annoted_captions = []
        for caption, inst_desc_mapping in zip(caption_tokens, ann['inst_desc_map']):
            caption = copy.copy(caption)
            for ((l, r), _) in inst_desc_mapping:
                if masked:
                    caption[l:r] = ['_' for _ in range(r-l)]
                else:
                    caption[l] = '[TL] ' + caption[l]
                    caption[r-1] = caption[r-1] + ' [TR]'
            annoted_captions.append(caption)

        all_template_captions[ann_catname] = (annoted_captions, color)
    
    all_caption_images = []
    for k, (template_captions, color) in all_template_captions.items():
        template_captions = [' '.join(x) for x in template_captions]
        caption_image = draw_captions(image.size[1], template_captions, title=k+':', color=color)
        all_caption_images.append(caption_image)
    final_image = image_hcat([image] + all_caption_images)
    return final_image

def draw_instance(image, ann, min_height=200, bbox=False):
    """draw instance image: image_hcat([instance(croped), descriptios])
    
    Args:
        masked (Bool): whether to mask out descriptions corresponding to instances
    """
    x1, y1, x2, y2 = bbox2xy(ann['bbox'])
    region = image.crop((x1, y1, x2, y2))
    mask = ann['mask'][y1:y2, x1:x2]
    if bbox:
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

def save_idmanns(idmanns):
    pass

def load_idmanns(idmanns):
    pass

if __name__ == '__main__':
    config = OmegaConf.load('configs/idm.yaml')
    idmAnns = inst_desc_match(config)

    import ipdb; ipdb.set_trace()