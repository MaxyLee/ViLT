import os
import pickle
import pandas as pd
import pyarrow as pa

from tqdm import tqdm

def path2rest(root, filename, split):
    with open(f'{root}/CUB_200_2011/images/{filename}.jpg', 'rb') as fp:
        binary = fp.read()

    with open(f'{root}/text/{filename}.txt', 'r') as fp:
        captions = fp.read().splitlines()

    return [binary, captions, filename, split]

def make_arrow(root):
    dataset_root = f'{root}/cub_arrows'
    with open(f'{root}/train/filenames.pickle', 'rb') as fp:
        train_filenames = pickle.load(fp)
    with open(f'{root}/test/filenames.pickle', 'rb') as fp:
        val_test_filenames = pickle.load(fp)

    val_filenames = []
    test_filenames = []
    for i, fn in enumerate(val_test_filenames):
        if i & 1 == 0:
            val_filenames.append(fn)
        else:
            test_filenames.append(fn)

    splits = ['train', 'val', 'test']
    all_filenames = [train_filenames, val_filenames, test_filenames]

    for split, filenames in zip(splits, all_filenames):
        bs = [path2rest(root, fn, split) for fn in tqdm(filenames, desc=split)]
        dataframe = pd.DataFrame(
            bs, columns=["image", "caption", "filename", "split"],
        )
        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/cub_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        print(f'Wrote {len(filenames)} images to {split} set')