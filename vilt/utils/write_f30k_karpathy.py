import json
import pandas as pd
import pyarrow as pa
import random
import os
import nlpaug.augmenter.word as naw

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def make_cip(root, cip_root, name=None):
    bs = []

    with open(f'{cip_root}/captions.txt') as f:
        for line in tqdm(f):
            filepath, captions = line.split(':\t')
            filename = filepath.split('/')[-1]
            with open(f'{cip_root}/{filename}', "rb") as fp:
                binary = fp.read()
            captions = eval(captions)
            bs.append([binary, captions, filename, 'train'])

    dataframe = pd.DataFrame(
        bs, columns=["image", "caption", "image_id", "split"],
    )

    table = pa.Table.from_pandas(dataframe)
    os.makedirs(root, exist_ok=True)
    fn = f'f30k_train_cip_{name}.arrow' if name else 'f30k_train_cip.arrow'
    with pa.OSFile(f"{root}/{fn}", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

def back_translation(root):
    back_translation_aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en', device='cuda:2')

    with open(f"{root}/karpathy/dataset_flickr30k.json", "r") as fp:
        captions = json.load(fp)

    captions = captions["images"]

def path2rest(path, iid2captions, iid2split):
    name = path.split("/")[-1]

    with open(path, "rb") as fp:
        binary = fp.read()

    captions = iid2captions[name]
    split = iid2split[name]

    return [binary, captions, name, split]


def make_arrow(root, dataset_root):
    with open(f"{root}/karpathy/dataset_flickr30k.json", "r") as fp:
        captions = json.load(fp)

    captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()

    for cap in tqdm(captions):
        filename = cap["filename"]
        iid2split[filename] = cap["split"]
        for c in cap["sentences"]:
            iid2captions[filename].append(c["raw"])

    paths = list(glob(f"{root}/flickr30k-images/*.jpg"))
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

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]

        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/f30k_caption_karpathy_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


if __name__ == '__main__':
    root = '/data2/share/data/ViLT/data/flickr30k'
    cip_root = '/data/private/mxy/projects/mmda/code/ViLT/CIP/tmp/small-dataset-tree/augment_results'
    # cip_root = '/data2/private/cc/experiment/ViLT/CIP/tmp/small-dataset-rm_small_obj/augment_results'
    make_cip(root, cip_root, name='tree')