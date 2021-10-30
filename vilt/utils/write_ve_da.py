import json
import pandas as pd
import pyarrow as pa
import re
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from .glossary import normalize_word

label_dict = {
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2,
}

img_pattern = re.compile(r"[0-9]*?.jpg")

def path2rest(root, split, img_name, row):
    texts = [r['sentence2'] for r in row if r['gold_label'] != '-']
    labels = [[label_dict[r['gold_label']]] for r in row if r['gold_label'] != '-']
    scores = [[1.0]] * len(labels)

    with open(f"{root}/flickr30k-images/{img_name}", "rb") as fp:
        image = fp.read()

    return [image, texts, labels, scores, split]

def path2rest_da(img_name, text, split):
    texts = [text]
    labels = [[1]]
    scores = [[1.0]]

    with open(img_name, "rb") as fp:
        image = fp.read()

    return [image, texts, labels, scores, split]


def make_arrow(root, txt_root, img_root, dataset_root):
    # synthetic data only for training
    split = 'train'

    # write real data
    with open(f"{root}/snli_1.0/snli_1.0_train.jsonl", "r") as fp:
        snli_train = [json.loads(jline) for jline in fp]

    annot = defaultdict(list)
    for row in tqdm(snli_train):
        img_fname = row['captionID'].split('#')[0]
        if not img_pattern.match(img_fname):
            print(f"Error image name: {img_fname}")
            continue
        annot[img_fname].append(row)

    bs = [
        path2rest(root, split, img_name, row) for img_name, row in tqdm(annot.items())
    ]

    # write synthetic data
    with open(txt_root + '/f30k_filenames.txt', 'r') as fin:
        filenames = fin.readlines()

    img_names = []
    texts = []
    for i, filename in enumerate(filenames):
        fname = filename.strip('\n')
        with open(f'{txt_root}/{fname}.txt') as fin:
            txts = fin.readlines()
        for j, txt in enumerate(txts):
            img_name = f'{img_root}/{i}/0_s_{j}_g2.png'
            if os.path.isfile(img_name):
                img_names.append(img_name)
                texts.append(txt.strip('\n'))
    
    print(f'da size: {len(texts)}')

    bs_da = [
        path2rest_da(img_name, text, split) for img_name, text in tqdm(zip(img_names, texts))
    ]

    dataframe = pd.DataFrame(
        bs + bs_da,
        columns=[
            "image",
            "text",
            "labels",
            "scores",
            "split",
        ]
    )

    table = pa.Table.from_pandas(dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(f"{dataset_root}/ve_{split}_da.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)