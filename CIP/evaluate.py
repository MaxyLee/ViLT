import json
from sys import base_prefix
import numpy as np

from tqdm import tqdm
from clip_score import CLIPScore

def origin_clip_scores():
    caption_file = '/data2/share/data/coco/annotations/small_dataset/captions_train.json'
    with open(caption_file) as f:
        captions = json.load(f)
    captions = captions['images']

    clip_score = CLIPScore(device='cuda:1')

    base_path = '/data2/share/data/coco/images/train2017'
    scores = {}
    for img in tqdm(captions):
        img_path = f"{base_path}/{img['file_name']}"

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
    origin_clip_scores()