import json
import stanza

from tqdm import tqdm
from pycocotools.coco import COCO
from collections import defaultdict

from coco_utils import loadCaptions

def run_extract_tree_refcoco(config, output_dir=None):
    import sys
    sys.path.append('/data1/private/mxy/projects/mmda/code/refer')
    from refer import REFER

    print('[Run]: constituency tree extraction')

    dataroot = config['dataroot']
    dataset = config['dataset']
    splitby = config['splitby']

    refer = REFER(dataroot, dataset, splitby)
    refs = refer.Refs
    
    stanza_dir = config['stanza_dir']
    nlp = stanza.Pipeline(lang='en', dir=stanza_dir, processors='tokenize,pos,constituency', use_gpu=False)

    id2tree = defaultdict(dict)
    for refid, ref in tqdm(refs.items(), desc='Extract Constituency Tree'):
        captions = [s['sent'] for s in ref['sentences']]
        for caption in captions:
            parsed_result = nlp(caption).sentences[0]
            tree = parsed_result.constituency
            id2tree[refid][caption] = str(tree)
    
    output_dir = output_dir or config['output_dir']
    with open(f'{output_dir}/id2tree_{dataset}.json', 'w') as f:
        json.dump(id2tree, f)

def run_extract_tree(config, output_dir=None):
    print('[Run]: constituency tree extraction')

    caption_anns = COCO(config['caption_annotation'])
    instance_anns = COCO(config['instance_annotation'])

    imgIds = instance_anns.getImgIds()
    
    stanza_dir = config['stanza_dir']
    nlp = stanza.Pipeline(lang='en', dir=stanza_dir, processors='tokenize,pos,constituency')

    id2tree = defaultdict(dict)
    for imgid in tqdm(imgIds, desc='Extract Constituency Tree'):
        captions = loadCaptions(caption_anns, imgid)
        for caption in captions:
            parsed_result = nlp(caption).sentences[0]
            tree = parsed_result.constituency
            id2tree[imgid][caption] = str(tree)
    
    output_dir = output_dir or config['output_dir']
    with open(f'{output_dir}/id2tree.json', 'w') as f:
        json.dump(id2tree, f)
