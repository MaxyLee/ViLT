import json
import pandas as pd
import pyarrow as pa
import os
import nlpaug.augmenter.word as naw

from tqdm import tqdm
from collections import defaultdict

def back_translation(root):
    back_translation_aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en', device='cuda:2')

    train_data = list(
        map(json.loads, open(f"{root}/nlvr2/data/train.json").readlines())
    )

    with open(f'{root}/nlvr2/data/train_da.json', 'w') as fout:
        data = []
        for l in tqdm(train_data, desc='back translation'):
            data.append(l)
            if len(data) == 21:
                texts = [d['sentence'] for d in data]
                da_texts = back_translation_aug.augment(texts)
                for i, d in enumerate(data):
                    d.update({'sentence_da': da_texts[i]})
                    json.dump(d, fout)
                    fout.write('\n')
                data = []


def process(root, iden, row):
    texts = [r["sentence"] for r in row]
    da_texts = [r["sentence_da"] for r in row]
    labels = [r["label"] for r in row]

    split = iden.split("-")[0]

    if iden.startswith("train"):
        directory = row[0]["directory"]
        path = f"{root}/images/train/{directory}/{iden}"
    else:
        path = f"{root}/{split}/{iden}"

    with open(f"{path}-img0.png", "rb") as fp:
        img0 = fp.read()
    with open(f"{path}-img1.png", "rb") as fp:
        img1 = fp.read()

    return [img0, img1, texts, da_texts, labels, iden]


def make_arrow(root, dataset_root):
    train_data = list(
        map(json.loads, open(f"{root}/nlvr2/data/train_da.json").readlines())
    )
    test1_data = list(
        map(json.loads, open(f"{root}/nlvr2/data/test1.json").readlines())
    )
    dev_data = list(map(json.loads, open(f"{root}/nlvr2/data/dev.json").readlines()))

    balanced_test1_data = list(
        map(
            json.loads,
            open(f"{root}/nlvr2/data/balanced/balanced_test1.json").readlines(),
        )
    )
    balanced_dev_data = list(
        map(
            json.loads,
            open(f"{root}/nlvr2/data/balanced/balanced_dev.json").readlines(),
        )
    )

    unbalanced_test1_data = list(
        map(
            json.loads,
            open(f"{root}/nlvr2/data/unbalanced/unbalanced_test1.json").readlines(),
        )
    )
    unbalanced_dev_data = list(
        map(
            json.loads,
            open(f"{root}/nlvr2/data/unbalanced/unbalanced_dev.json").readlines(),
        )
    )

    splits = [
        "train",
        # "dev",
        # "test1",
        # "balanced_dev",
        # "balanced_test1",
        # "unbalanced_dev",
        # "unbalanced_test1",
    ]

    datas = [
        train_data,
        dev_data,
        test1_data,
        balanced_dev_data,
        balanced_test1_data,
        unbalanced_dev_data,
        unbalanced_test1_data,
    ]

    annotations = dict()

    for split, data in zip(splits, datas):
        _annot = defaultdict(list)
        for row in tqdm(data):
            _annot["-".join(row["identifier"].split("-")[:-1])].append(row)
        annotations[split] = _annot

    for split in splits:
        bs = [
            process(root, iden, row) for iden, row in tqdm(annotations[split].items())
        ]

        dataframe = pd.DataFrame(
            bs, columns=["image_0", "image_1", "questions", "questions_da", "answers", "identifier"],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/nlvr2_{split}_da.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
