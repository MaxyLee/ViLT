import os, random
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import scale
import numpy as np

from tqdm import tqdm

train_paths = ['/data2/share/data/coco/images/train2017', 
               '/data2/share/data/coco/annotations/captions_train2017.json', 
               '/data2/share/data/coco/annotations/instances_train2017.json',
               '']

val_paths = ['/data2/share/data/coco/images/val2017', 
             '/data2/share/data/coco/annotations/captions_val2017.json', 
             '/data2/share/data/coco/annotations/instances_val2017.json',
             '/data2/private/cc/experiment/ViLT/demo/coco_viz/samples/dog-inpaint-output']


def load_anns(split='val'):
    if split == 'train':
        paths = train_paths
    else:
        paths = val_paths
    return {
        'image_path': paths[0], 
        'caption_anns': COCO(paths[1]), 
        'instance_anns': COCO(paths[2]),
        'inpaint_image_path': paths[3]
    }

def bbox2xy(bbox):
    x, y, w, h = bbox
    x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
    x2, y2 = max(x2, x1+1), max(y2, y1+1)
    return x1, y1, x2, y2

def image_id2path(image_path, imgid):
    return '{image_path}/{imgid:0>12d}.jpg'

def loadImage(image_path, imgid, fmt='{imgid:0>12d}.jpg'):
    formatted_fn = fmt.format(imgid=imgid)
    im = io.imread(f'{image_path}/{formatted_fn}')
    im = Image.fromarray(im)
    return im

def loadCaptions(caption_anns, imgid):
    captions = caption_anns.loadAnns(caption_anns.getAnnIds(imgIds=[imgid]))
    captions = [cap['caption'] for cap in captions]
    return captions

def pad_2darray(array, height=None, width=None, mode='center'):
    h, w = array.shape[:2]
    pad_h, pad_w = (0,0), (0,0)
    if height is not None:
        pt = (height-h) // 2
        pb = height - h - pt
        pad_h = (pt, pb)
    if width is not None:
        pl = (width-w) // 2
        pr = width - w - pl
        pad_w = (pl, pr)
    return np.pad(array, (pad_h, pad_w, (0,0)), 'constant')

def image_hcat(images):
    max_height = max([image.size[1] for image in images])
    return Image.fromarray(np.concatenate([pad_2darray(np.array(img), height=max_height) for img in images], axis=1))

def image_vcat(images):
    max_width = max([image.size[0] for image in images])
    return Image.fromarray(np.concatenate([pad_2darray(np.array(img), width=max_width) for img in images], axis=0))


def draw_instance_bbox(image, bbox, color='red', label='bbox'):
    """show instance bbox in the image
    """
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

def draw_image_captions(image, captions):
    caption_image = draw_captions(image.size[1], captions)
    return image_hcat([image, caption_image])