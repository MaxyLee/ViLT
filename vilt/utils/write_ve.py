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


def make_arrow(root, dataset_root):
    with open(f"{root}/snli_1.0/snli_1.0_train.jsonl", "r") as fp:
        snli_train = [json.loads(jline) for jline in fp]
    with open(f"{root}/snli_1.0/snli_1.0_dev.jsonl", "r") as fp:
        snli_val = [json.loads(jline) for jline in fp]
    with open(f"{root}/snli_1.0/snli_1.0_test.jsonl", "r") as fp:
        snli_test = [json.loads(jline) for jline in fp]

    splits = ['train', 'val', 'test']
    datas = [snli_train, snli_val, snli_test]

    annotations = dict()

    for split, data in zip(splits, datas):
        _annot = defaultdict(list)
        for row in tqdm(data):
            img_fname = row['captionID'].split('#')[0]
            if not img_pattern.match(img_fname):
                print(f"Error image name: {img_fname}")
                continue
            _annot[img_fname].append(row)
            annotations[split] = _annot

    for split in splits:
        bs = [
            path2rest(root, split, img_name, row) for img_name, row in tqdm(annotations[split].items())
        ]

        dataframe = pd.DataFrame(
            bs,
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
        with pa.OSFile(f"{dataset_root}/ve_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)