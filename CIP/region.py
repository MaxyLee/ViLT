import os, random
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image
from torchvision.transforms.functional import scale
import numpy as np

from tqdm import tqdm
from clip_score import CLIPScore

train_paths = ['/data2/share/data/coco/images/train2017', 
               '/data2/share/data/coco/annotations/captions_train2017.json', 
               '/data2/share/data/coco/annotations/instances_train2017.json']

val_paths = ['/data2/share/data/coco/images/val2017', 
               '/data2/share/data/coco/annotations/captions_val2017.json', 
               '/data2/share/data/coco/annotations/instances_val2017.json']

def bbox2xy(bbox):
    x, y, w, h = bbox
    x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
    x2, y2 = max(x2, x1+1), max(y2, y1+1)
    return x1, y1, x2, y2

def loadImage(image_path, imgid):
    im = io.imread(f'{image_path}/{imgid:0>12d}.jpg')
    im = Image.fromarray(im)
    return im

def loadCaptions(caption_anns, imgid):
    captions = caption_anns.loadAnns(caption_anns.getAnnIds(imgIds=[imgid]))
    captions = [cap['caption'] for cap in captions]
    return captions

def getRegion(image_path, ann, return_mask=False):
    image_id = ann['image_id']
    image = loadImage(image_path, image_id)
    x1, y1, x2, y2 = bbox2xy(ann['bbox'])
    region = image.crop((x1, y1, x2, y2))
    mask = ann['mask'][y1:y2, x1:x2]
    if return_mask:
        return region, mask
    return region

def replaceRegion(image_path, src_ann, tgt_ann, image=None):
    src_bbox = bbox2xy(src_ann['bbox'])
    src_size = (src_bbox[2]-src_bbox[0], src_bbox[3]-src_bbox[1]) # x2 - x1, y2 - y1
    if image is None:
        image = loadImage(image_path, src_ann['image_id'])

    tgt_region, tgt_mask = getRegion(image_path, tgt_ann, return_mask=True)
    tgt_mask = Image.fromarray(tgt_mask*255)
    tgt_mask = tgt_mask.resize(src_size)

    tgt_region = tgt_region.resize(src_size) 
    image.paste(tgt_region, box=src_bbox, mask=tgt_mask)
    return image

def is_match(src_ann, tgt_ann):
    src_x, src_y, src_w, src_h = src_ann['bbox']
    tgt_x, tgt_y, tgt_w, tgt_h = tgt_ann['bbox']
    scale_w, scale_h = src_w / tgt_w, src_h / tgt_h
    if 0.5 < scale_w and 0.5 < scale_h:
        return True
    return False

def augment_once(instance_ann, src_ann, origin_image, max_try=50):
    image = origin_image.copy()

    # tgt_categories = [src_ann['category_id']]
    tgt_categories = instance_ann.getCatIds(catNms=['sheep'])

    tgt_ann_id = random.choice(instance_ann.getAnnIds(catIds=tgt_categories))
    tgt_ann = instance_ann.loadAnns([tgt_ann_id])[0]
    try_times = 1
    while (not is_match(src_ann, tgt_ann)):
        tgt_ann_id = random.choice(instance_ann.getAnnIds(catIds=tgt_categories))
        tgt_ann = instance_ann.loadAnns([tgt_ann_id])[0]
        try_times += 1
        if try_times >= max_try:
            return None, None
    
    src_ann['mask'] = instance_ann.annToMask(src_ann)
    tgt_ann['mask'] = instance_ann.annToMask(tgt_ann)
    augmented_image = replaceRegion(image_path, src_ann, tgt_ann, image=image)
    return augmented_image, tgt_ann_id

def loadImage_mask(image_path, imgid):
    im = io.imread(f'{image_path}/{imgid}_mask.png')
    im = Image.fromarray(im)
    return im

if __name__ == '__main__':
    # instances_train_coco = COCO(train_paths[2])
    # cats = instances_train_coco.loadCats(instances_train_coco.getCatIds())

    image_path = val_paths[0]
    caption_anns = COCO(val_paths[1])
    coco = COCO(val_paths[2])

    # Categories

    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # nms = set([cat['supercategory'] for cat in cats])
    # print('COCO supercategories: \n{}'.format(' '.join
    # (nms)))

    # clip_score = CLIPScore(device='cpu')

    nms = ['dog']
    output_dir = 'samples/dog2sheep-inpaint'
    os.makedirs(output_dir, exist_ok=True)

    score_fout = open(f'{output_dir}/clip_score.txt', 'w')

    for nm in tqdm(nms):
        catIds = coco.getCatIds(catNms=[nm])
        imgIds = coco.getImgIds(catIds=catIds)

        for imgId in tqdm(imgIds):
            origin_image = loadImage(image_path, imgId)
            annIds = coco.getAnnIds(imgIds=[imgId], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)

            captions = loadCaptions(caption_anns, imgId)

            src_ann = anns[0]

            ############## inpaint ##############
            
            # mask = np.zeros((origin_image.size[1], origin_image.size[0]), dtype=np.uint8)
            # x1, y1, x2, y2 = bbox2xy(src_ann['bbox'])
            # mask[y1:y2, x1:x2] = 255
            # mask_image = Image.fromarray(mask)
            # origin_image.save(f'{output_dir}/{src_ann["image_id"]}.png')
            # mask_image.save(f'{output_dir}/{src_ann["image_id"]}_mask.png')

            # run lama

            import ipdb; ipdb.set_trace()
            
            ############## inpaint ##############
            
            mask_image = loadImage_mask('/data2/private/cc/experiment/ViLT/demo/coco_viz/samples/dog-inpaint-output', imgId)

            all_images = [origin_image]
            for i in range(5):
                augment_image, tgt_ann_id = augment_once(coco, src_ann, mask_image)
                if augment_image is None:
                    continue
                all_images.append(augment_image)
                augment_image.save(f'{output_dir}/{imgId}-{i}-{tgt_ann_id}.png')
            
            if len(all_images) == 1:
                continue

            origin_image.save(f'{output_dir}/{imgId}-origin.png')    
            scores = clip_score.score(captions, all_images, return_diag=False)
            score_fout.write(f'{imgId}:\n{captions}\n{scores}\n\n')

            
