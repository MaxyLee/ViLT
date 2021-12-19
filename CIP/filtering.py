import json

def run_filter(config):
    data_dir = config['data_dir']
    clip_th = config['clip_th']

    total_num = 0
    filtered_filenames = []

    with open(f'{data_dir}/scores.txt', 'r') as fin:
        for line in fin:
            filename, score = line.split(':\t')
            score = float(score.split('\t')[0])
            if score >= clip_th:
                filtered_filenames.append(filename)
            total_num += 1

    print(f'Get {len(filtered_filenames)} images from {total_num}({len(filtered_filenames)/total_num*100.0:.2f}%)')

    with open(f'{data_dir}/captions.txt', 'r') as fin:
        with open(f'{data_dir}/filtered_captions.txt', 'w') as fout:
            for line in fin:
                filename = line.split(':\t')[0]
                if filename in filtered_filenames:
                    fout.write(line)
    