import os
import json
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict

import numpy as np
from PIL import Image
from nltk.tree import Tree
from nltk.draw.tree import TreeView
from pycocotools.coco import COCO
import torch
import torchvision.transforms as T
torch.set_grad_enabled(False)

from graphviz import Digraph

import stanza

from coco_utils import (
    loadCaptions, loadImage
)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def inst_match(src_inst, tgt_inst, config):
    """
    """
    src_mask, tgt_mask = src_inst['mask'], tgt_inst['mask']

    src_size = (src_mask.shape[1], src_mask.shape[0])
    tgt_size = (tgt_mask.shape[1], tgt_mask.shape[0])
    scale_w, scale_h = src_size[0] / tgt_size[0], src_size[1] / tgt_size[1]

    # too small
    hw_scale = config.get('th_hw_scale', 0.7)
    if scale_w < hw_scale or scale_h < hw_scale:
        return False
    
    # aspect ratio gap is too large
    aspect_ratio_src = src_size[0] / src_size[1]
    aspect_ratio_tgt = tgt_size[0] / tgt_size[1]
    th_aspect_ratio = config.get('th_aspect_ratio', 0.7)
    if abs(aspect_ratio_src-aspect_ratio_tgt) > th_aspect_ratio:
        return False

    # shape similarity (src_size => tgt_size)
    src_mask = Image.fromarray(src_mask)
    src_mask = np.array(src_mask.resize(tgt_size))
    sim = (src_mask * tgt_mask).sum() / ((src_mask+tgt_mask) != 0).sum()
    if sim < config.get('th_areascore', 0.7):
        return False

    # VBG comparing
    return True

def phrase_grounding(image, caption, model):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(image).unsqueeze(0).cuda()

    # propagate through the model
    memory_cache = model(img, [caption], encode_and_save=True)
    outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

    # keep only predictions with 0.7+ confidence
    probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = (probas > 0.7).cpu()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], image.size)

    # Extract the text spans predicted by each box
    positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
    predicted_spans = defaultdict(str)
    for tok in positive_tokens:
        item, pos = tok
        if pos < 255:
            span = memory_cache["tokenized"].token_to_chars(0, pos)
            predicted_spans [item] += " " + caption[span.start:span.end]

    labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
    return probas[keep], bboxes_scaled, labels

def draw_tree(constituency_tree, fn='graph'):
    graph = Digraph(format="png")
    
    nodes = []
    def visit(node, current_idx, g):
        name = node.label
        index = len(nodes)
        nodes.append(node)
        if node.is_leaf():
            g.node(str(index), name, color='lightblue2', shape="rectangle")
        else:
            g.node(str(index), name)
        if index != current_idx:
            g.edge(str(current_idx), str(index))
        if not node.is_leaf():
            for child_node in node.children:
                visit(child_node, index, g)

    nodes.append(constituency_tree)
    graph.node('0', constituency_tree.label)
    visit(constituency_tree.children[0], 0, graph)
    rendered_imgfn = graph.render(fn)
    return rendered_imgfn


def test_inst_desc_match(config):
    output_dir = config['output_dir']

    image_path = config['image_path']
    caption_anns = COCO(config['caption_annotation'])
    instance_anns = COCO(config['instance_annotation'])

    # MDETR model
    # model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=False, return_postprocessor=True)
    # ckpt = torch.load('checkpoints/pretrained_EB5_checkpoint.pth')
    # model.load_state_dict(ckpt['model'])
    # model = model.cuda()
    # model.eval()
    
    # Stanza
    nlp = stanza.Pipeline(lang='en', dir='/data2/private/cc/experiment/ViLT/CIP/checkpoints/stanza', processors='tokenize,pos,constituency')

    os.makedirs(output_dir, exist_ok=True)

    imgIds = instance_anns.getImgIds()
    idmAnns = []
    
    
    for imgid in tqdm(imgIds[:1000], desc='Parse Templates'):
        image = loadImage(image_path, imgid)
        image.save(f'{output_dir}/{imgid}.png')
        # instance-description alignment for a single image
        captions = loadCaptions(caption_anns, imgid)

        instance_annIds = instance_anns.getAnnIds(imgIds=[imgid])
        anns = instance_anns.loadAnns(instance_annIds)

        img_size = (instance_anns.imgs[imgid]['height'], instance_anns.imgs[imgid]['width'])
        img_area = img_size[0] * img_size[1]

        # filter out unsuitable instances
        filtered_anns = []
        for ann in anns:
            ann_catid = ann['category_id']
            inst_area = ann['bbox'][-2] * ann['bbox'][-1]
            if (float(inst_area) / img_area) < config.get('th_coverage', 0.1):
                continue
            filtered_anns.append(ann)
        anns = filtered_anns

        inst_desc_mapping = {}
        caption_tokens = []

        for idx, caption in enumerate(captions):
            # phrase grounding
            # probs, bboxes, labels = phrase_grounding(image, caption, model)

            # sentence constituency parsing
            parsed_result = nlp(caption).sentences[0]
            constituency_tree = parsed_result.constituency

            # draw_tree(constituency_tree, fn=f'{output_dir}/{imgid}-{idx}')

            import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    config = OmegaConf.load('configs/config.yaml')['experimental']
    test_inst_desc_match(config['inst_desc_match'])