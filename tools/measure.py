import os
import json
import numpy as np

from tqdm import tqdm

RESULT_PATH = '/share/nas165/amian/experiments/nlp/DPR/outputs/2022-07-09/00-01-39/result.json'

def read_json(path):
    datas = None
    with open(path, 'r', encoding='utf-8') as fr:
        datas = json.load(fr)
    return datas

result_datas = read_json(RESULT_PATH)

def recall(topk, datas):
    hits = 0
    for data in datas:
        ctxs = data['ctxs']
        hit   = [int(ctx['has_answer']) for ctx in ctxs]
        hits += 1 if sum(hit[:topk]) > 0 else 0
    return hits / len(datas)

for k in [1, 5, 10]:
    score = recall(k, result_datas)
    print('Hits@{}: {}'.format(k, np.round(score*100, 2)))