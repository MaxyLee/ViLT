from genericpath import exists
import os
import json
import torch
import requests
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from collections import defaultdict
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage.measure import find_contours

from inst_desc_match import IDMAnn, save_idmanns
from f30k_utils import get_sentence_data, get_annotations
from coco_utils import draw_captions, image_vcat, loadCaptions, loadImage, draw_instance_bbox

torch.set_grad_enabled(False)

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

def box_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x+w, y+h]

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

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def plot_results_plt(pil_img, scores, boxes, labels, masks=None, fn='result'):
    plt.figure(figsize=(16,10))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
      masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        if mask is None:
          continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          # Subtract the padding and flip (y, x) to (x, y)
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none", edgecolor=c)
          ax.add_patch(p)


    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(fn)

def plot_results(pil_img, scores, boxes, labels, masks=None, caption=None, fn='result'):
    # plt.figure(figsize=(16,10))
    # np_image = np.array(pil_img)
    # ax = plt.gca()
    img = pil_img.copy()
    colors = COLORS * 100
    if masks is None:
        masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        im = draw_instance_bbox(img, [xmin, ymin, xmax-xmin, ymax-ymin], label=f"{l}: {s}")
        # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                            fill=False, color=c, linewidth=3))
        # text = f'{l}: {s:0.2f}'
        # ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        # if mask is None:
        #     continue
        # np_image = apply_mask(np_image, mask, c)

        # padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        # padded_mask[1:-1, 1:-1] = mask
        # contours = find_contours(padded_mask, 0.5)
        # for verts in contours:
        #     # Subtract the padding and flip (y, x) to (x, y)
        #     verts = np.fliplr(verts) - 1
        #     p = Polygon(verts, facecolor="none", edgecolor=c)
        #     ax.add_patch(p)

    if caption:
        # im = Image.fromarray(np_image)
        cap_im = draw_captions(30, [caption])
        im = image_vcat([im, cap_im])
        # np_image = np.asarray(im)

    im.save(fn)
    # plt.imshow(np_image)
    # plt.axis('off')
    # plt.savefig(fn)


def add_res(results, ax, color='green'):
    #for tt in results.values():
    if True:
        bboxes = results['boxes']
        labels = results['labels']
        scores = results['scores']
        #keep = scores >= 0.0
        #bboxes = bboxes[keep].tolist()
        #labels = labels[keep].tolist()
        #scores = scores[keep].tolist()
    #print(torchvision.ops.box_iou(tt['boxes'].cpu().detach(), torch.as_tensor([[xmin, ymin, xmax, ymax]])))
    
    colors = ['purple', 'yellow', 'red', 'green', 'orange', 'pink']
    
    for i, (b, ll, ss) in enumerate(zip(bboxes, labels, scores)):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=colors[i], linewidth=3))
        cls_name = ll if isinstance(ll,str) else CLASSES[ll]
        text = f'{cls_name}: {ss:.2f}'
        print(text)
        ax.text(b[0], b[1], text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

def inference(im, caption, model):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).cuda()

    # propagate through the model
    memory_cache = model(img, [caption], encode_and_save=True)
    outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

    # keep only predictions with 0.7+ confidence
    probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = (probas > 0.7).cpu()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

    # Extract the text spans predicted by each box
    positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
    predicted_spans = defaultdict(str)
    for tok in positive_tokens:
        item, pos = tok
        if pos < 255:
            span = memory_cache["tokenized"].token_to_chars(0, pos)
            predicted_spans[item] += " " + caption[span.start:span.end]

    labels = [predicted_spans[k] for k in sorted(list(predicted_spans.keys()))]

    return probas[keep], bboxes_scaled, labels

def plot_inference(im, caption, fn):
    scores, boxes, labels = inference(im, caption)
    plot_results(im, scores, boxes, labels, caption=caption, fn=fn)

def inference_segmentation(im, caption, model_pc):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).cuda()

    # propagate through the model
    outputs = model_pc(img, [caption])

    # keep only predictions with 0.9+ confidence
    probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = (probas > 0.9).cpu()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

    # Interpolate masks to the correct size
    w, h = im.size
    masks = F.interpolate(outputs["pred_masks"], size=(h, w), mode="bilinear", align_corners=False)
    masks = masks.cpu()[0, keep].sigmoid() > 0.5

    tokenized = model_pc.detr.transformer.tokenizer.batch_encode_plus([caption], padding="longest", return_tensors="pt").to(img.device)

    # Extract the text spans predicted by each box

    positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
    predicted_spans = defaultdict(str)
    for tok in positive_tokens:
        item, pos = tok
        if pos < 255:
            try:
                span = tokenized.token_to_chars(0, pos)
                predicted_spans [item] += " " + caption[span.start:span.end]
            except:
                print('whoops')


    labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
    return probas[keep], bboxes_scaled, labels, masks

def plot_inference_segmentation(im, caption):
    scores, boxes, labels, masks = inference_segmentation(im, caption)
    plot_results(im, scores, boxes, labels, masks)
    # return outputs

def get_bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def get_overlap(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)

def run_phrase_grounding(config, output_dir=None):
    print('[Run]: phrase grounding')

    # load mdetr model
    print('loading MDETR model...')
    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=False, return_postprocessor=True)
    ckpt = torch.load('checkpoints/pretrained_EB5_checkpoint.pth')
    model.load_state_dict(ckpt['model'])
    model = model.cuda()
    model.eval()

    th = config['th_bbox_overlap']
    image_path = config['image_path']
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

    idmAnns = []
    for imgid in tqdm(imgIds[:50], desc='Parse Templates'):
        image = loadImage(image_path, imgid)
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

        folder_name = f"tmp/mdetr/pg-{imgid}"
        os.makedirs(folder_name, exist_ok=True)
        for i, caption in enumerate(captions):
            caption = caption.lower()
            scores, boxes, labels = inference(image, caption, model)
            if not len(scores) == len(labels):
                import ipdb; ipdb.set_trace()
            
            # match annotation bboxes with predicted bboxes
            for j, ann in enumerate(anns):
                ann_id = ann['id']
                ann_bbox = box_xywh_to_xyxy(ann['bbox'])
                ann_catid = ann['category_id']
                ann_catname = instance_anns.cats[ann_catid]['name']
                ann_caption_mapping = []
                
                # calculate overlap ratio for matching
                for k, pred_bbox in enumerate(boxes):
                    ann_area = get_bbox_area(ann_bbox)
                    pred_area = get_bbox_area(pred_bbox)
                    overlap_area = get_overlap(ann_bbox, pred_bbox)

                    ann_ratio = overlap_area / ann_area
                    pred_ratio = overlap_area / pred_area
                    iou = overlap_area / (ann_area + pred_area - overlap_area)

                    fn = f'{folder_name}/{i}-{j}-{k}.png'
                    test_scores = torch.Tensor([ann_ratio, pred_ratio])
                    test_boxes = torch.Tensor([ann_bbox , pred_bbox])
                    test_labels = [f"ann:{ann_catname}", f"pred:{labels[k]}"]
                    plot_results(image, test_scores, test_boxes, test_labels, caption=f'{caption} iou:{round(iou, 4)}', fn=fn)


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

    import ipdb; ipdb.set_trace()
    # if output_dir is not None:
    #     os.makedirs(output_dir, exist_ok=True)
    #     save_idmanns(idmAnns, config, output_dir=output_dir)
    # elif 'output_dir' in config:
    #     output_dir = config['output_dir']
    #     os.makedirs(output_dir, exist_ok=True)
    #     save_idmanns(idmAnns, config, output_dir=output_dir)
    return idmAnns

def run_segmentation(config, output_dir=None):
    root = config['root']
    output_dir = output_dir or config['output_dir']
    img_path = f'{root}/flickr30k-images'
    ann_path = f'{root}/Annotations'
    cap_path = f'{root}/Sentences'

    with open(f'{root}/karpathy/dataset_flickr30k.json', 'r') as f:
        dataset = json.load(f)
    images = dataset['images']

    imgids = [img['filename'][:-4] for img in images if img['split'] == 'train']

    # load model
    model_pc = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB3_phrasecut', pretrained=False, return_postprocessor=False)
    ckpt = torch.load('checkpoints/phrasecut_EB3_checkpoint.pth')
    model_pc.load_state_dict(ckpt['model'])
    model_pc = model_pc.cuda()
    model_pc.eval()

    imgids = ['4797050581', '3923857105', '2652311904']
    for imgid in tqdm(imgids):
        img = Image.open(f'{img_path}/{imgid}.jpg')
        anns = get_annotations(f'{ann_path}/{imgid}.xml')
        caps = get_sentence_data(f'{cap_path}/{imgid}.txt')

        segms = []
        phrase_ids = list(anns['boxes'].keys())
        for cap in caps:
            segm = {}
            for p in cap['phrases']:
                if p['phrase_id'] in phrase_ids:
                    scores, boxes, labels, masks = inference_segmentation(img, p['phrase'].lower(), model_pc)
                    segm[p['phrase_id']] = {
                        'scores': scores,
                        'boxes': boxes,
                        'labels': labels,
                        'masks': masks
                    }
            segms.append(segm)
        import ipdb; ipdb.set_trace()
                    # import ipdb; ipdb.set_trace()
                # plot_results_plt(img, scores, boxes, labels, masks, fn='tmp/test')
        # with open(f'{output_dir}/{imgid}.json', 'w') as f:
        #     json.dump(segm, f)
        torch.save(segm, f'{output_dir}/{imgid}')
                    


if __name__ == '__main__':
    print('[Run]: phrase grounding with MDETR')
    # load mdetr model
    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=False, return_postprocessor=True)
    ckpt = torch.load('checkpoints/pretrained_EB5_checkpoint.pth')
    model.load_state_dict(ckpt['model'])
    model = model.cuda()
    model.eval()

    # url = "http://images.cocodataset.org/val2017/000000281759.jpg"
    # im = Image.open(requests.get(url, stream=True).raw)
    # plot_inference(im, "5 people each holding an umbrella", 'tmp/mdetr/example')

    # load data
    root = '/data/share/data/coco'
    save_path = 'tmp/mdetr'
    caps = json.load(open(f'{root}/annotations/captions_train2017.json'))
    anns = caps['annotations']

    for ann in tqdm(anns[:50]):
        img_path = f"{root}/images/train2017/{str(ann['image_id']).zfill(12)}.jpg"
        im = Image.open(img_path)
        fn = f"{save_path}/{ann['image_id']}-{ann['id']}"
        # plot_inference(im, ann['caption'].lower(), fn)
        scores, boxes, labels = inference(im, ann['caption'].lower(), model)

        import ipdb; ipdb.set_trace()