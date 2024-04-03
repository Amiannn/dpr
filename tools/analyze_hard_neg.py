import os
import json

from tqdm import tqdm

NQ_TRAIN_SMALL_PATH = './downloads/data/retriever/nq-train-small-shardneg.json'

# SEMANTICIDS_PATH  = './tmp/psgs_w100_small_passage2semanticid_alphabet.tsv'
# PSGS_PATH = './downloads/data/wikipedia_split/psgs_w100_small_dpr.tsv'

def read_json(path):
    datas = None
    with open(path, 'r', encoding='utf-8') as fr:
        datas = json.load(fr)
    return datas

def compare_exist(negs, hard_negs):
    hit = 0
    for neg in negs:
        sid = neg['semantic_id']
        is_hit = False
        for hard_neg in hard_negs:
            hard_sid = hard_neg['semantic_id']
            if sid == hard_sid:
                is_hit = True
                break
        if is_hit:
            hit += 1
    # print('match: {}, length: {}'.format(hit, len(negs)))
    return hit / len(negs)

nq_train_small_datas = read_json(NQ_TRAIN_SMALL_PATH)

hits = 0
for data in nq_train_small_datas:
    negative_ctxs = data['negative_ctxs']
    hard_negative_ctxs = data['hard_negative_ctxs']
    hits += compare_exist(negative_ctxs, hard_negative_ctxs)

print('match: {}'.format(hits / len(nq_train_small_datas)))
