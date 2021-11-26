import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from pycocotools.coco import COCO
from collections import defaultdict

def make_subset(root, query_name='dog'):
    # load annotations
    ann_path = '/data/share/data/coco2017/annotations'
    train_ann = COCO(annotation_file=f'{ann_path}/instances_train2017.json')
    val_ann = COCO(annotation_file=f'{ann_path}/instances_val2017.json')

    query_id = val_ann.getCatIds(catNms=[query_name])[0]
    filter_ids = train_ann.getImgIds(catIds=[query_id]) + val_ann.getImgIds(catIds=[query_id])

    # load captions
    with open(f"{root}/karpathy/dataset_coco.json", "r") as fp:
        captions = json.load(fp)
    captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()
    
    cnts = defaultdict(int)
    for cap in captions:
        if cap['cocoid'] in filter_ids:
            cnts[cap['split']] += 1


    for cap in tqdm(captions):
        if cap['cocoid'] in filter_ids:
            filename = cap["filename"]
            iid2split[filename] = cap["split"]
            for c in cap["sentences"]:
                iid2captions[filename].append(c["raw"])

    paths = list(glob(f"{root}/train2014/*.jpg")) + list(glob(f"{root}/val2014/*.jpg"))
    random.shuffle(paths)
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

    print(f'file num: {len(caption_paths)}')

    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "restval", "test"]:
        batches = [b for b in bs if b[-1] == split]

        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(root, exist_ok=True)
        with pa.OSFile(
            f"{root}/coco_{split}_{query_name}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest(path, iid2captions, iid2split):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary, captions, name, split]


def make_arrow(root, dataset_root):
    with open(f"{root}/karpathy/dataset_coco.json", "r") as fp:
        captions = json.load(fp)

    captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()

    for cap in tqdm(captions):
        filename = cap["filename"]
        iid2split[filename] = cap["split"]
        for c in cap["sentences"]:
            iid2captions[filename].append(c["raw"])

    paths = list(glob(f"{root}/train2014/*.jpg")) + list(glob(f"{root}/val2014/*.jpg"))
    random.shuffle(paths)
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "restval", "test"]:
        batches = [b for b in bs if b[-1] == split]

        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/coco_caption_karpathy_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

if __name__ == '__main__':
    root = '/data/share/UNITER/origin_imgs/coco_original'
    make_subset(root)