from collections import defaultdict
import json
import random
import shutil
from sys import base_prefix
import numpy as np

from tqdm import tqdm
from clip_score import CLIPScore
from collections import defaultdict

def sample_images(inpath, outpath):
    random.seed(0)

    with open(f'{inpath}/captions.txt', 'r') as fin:
        lines = fin.readlines()

    sampled_lines = random.sample(lines, 100)

    with open(f'{outpath}/sampled_captions', 'w') as fout:
        for line in sampled_lines:
            image_fn, captions = line.strip().split(':\t')
            image_fn = image_fn.split('/')[-1]
            shutil.copyfile(f'{inpath}/{image_fn}', f'{outpath}/{image_fn}')
            fout.write(line)
        


def convert_caption_format(input, output):
    with open(input) as fin:
        dataset = json.load(fin)
    images = dataset['images']

    results = {}
    for img in images:
        fn = img['filename']
        results[fn] = [cap['raw'] for cap in img['sentences']]

    with open(output, 'w') as fout:
        for fn, captions in results.items():
            fout.write(f'/data2/share/data/flickr30k-entities/flickr30k-images/{fn}:\t{captions}\n')

def run_evaluate(config):
    print('[Run]: evaluate image-text pairs')
    clip_score = CLIPScore(device=config['device'])
    data_dir = config['data_dir']
    caption_file = f'{data_dir}/captions.txt'
    batch_size = config['batch_size']

    all_scores = []
    fout = open(config['output_path'], 'w')
    batch_images = []
    batch_captions = []
    batch_split = [0]
    for line in tqdm(open(caption_file)):
        image_fn, captions = line.strip().split(':\t')
        captions = eval(captions)

        batch_images.append(image_fn)
        batch_captions += captions
        batch_split.append(len(captions))
        if len(batch_images) == batch_size:
            scores = clip_score.score(batch_captions, batch_images, return_diag=False)
            batch_split = np.cumsum(batch_split)
            for i in range(batch_size):
                scores_i = scores[i][batch_split[i]:batch_split[i+1]].tolist()
                mean_score_i = np.mean(scores_i)
                all_scores += scores_i
                fout.write(f'{batch_images[i]}:\t{mean_score_i}\t{scores_i}\n')

            batch_images = []
            batch_captions = []
            batch_split = [0]

    if len(batch_images):
        scores = clip_score.score(batch_captions, batch_images, return_diag=False)
        batch_split = np.cumsum(batch_split)
        for i in range(len(batch_images)):
            scores_i = scores[i][batch_split[i]:batch_split[i+1]].tolist()
            mean_score_i = np.mean(scores_i)
            all_scores += scores_i
            fout.write(f'{batch_images[i]}:\t{mean_score_i}\t{scores_i}\n')

    print(f'Average CLIP-score: {np.mean(all_scores)}')

if __name__ == '__main__':
    # convert_caption_format('/data2/share/data/flickr30k-entities/karpathy/dataset_flickr30k.json', '/data2/share/data/flickr30k-entities/captions.txt')
    root = '/data1/private/mxy/projects/mmda/code/ViLT/CIP/tmp/coco'
    sample_images(f'{root}/augment_results', f'{root}/sampled_images')