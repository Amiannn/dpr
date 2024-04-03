import os
import json

from tqdm import tqdm

NQ_TRAIN_SMALL_PATH = './downloads/data/retriever/nq-train-small.json'

SEMANTICIDS_PATH  = './tmp/psgs_w100_small_passage2semanticid_alphabet.tsv'
PSGS_PATH = './downloads/data/wikipedia_split/psgs_w100_small_dpr.tsv'

def read_tsv(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for i, fr in tqdm(enumerate(frs)):
            if i == 0: continue
            datas.append((fr.replace('\n', '')).split('\t'))
    return datas

def read_json(path):
    datas = None
    with open(path, 'r', encoding='utf-8') as fr:
        datas = json.load(fr)
    return datas

def insert_leaf(tree, sid, value):
    root = tree
    now  = root
    for node_id in list(sid.replace(' ', '')):
        if node_id not in now:
            now[node_id] = {}
        now = now[node_id]
    now['value'] = value

def build_semantic_tree(semanticids):
    semantic_tree = {}
    for pid, sid in semanticids:
        insert_leaf(semantic_tree, sid, pid)
    return semantic_tree

def get_leaf(tree, sid, source_id):
    def walk(node):
        if 'value' in node: 
            return node['value'] 
        for _id in node:
            return walk(node[_id])

    now = tree
    for node_id in list(sid.replace(' ', '')):
       now = now[node_id]

    leafs = []
    for _id in now:
        if _id == source_id: continue
        leafs.append(walk(now[_id]))

    return leafs

def same_sub_cluster(tree, sid):
    sid = sid.replace(' ', '')
    leafs, scores = [], []
    for i in range(len(sid) - 1, 0, -1):
        _id = sid[:i]
        leaf = get_leaf(tree, _id, sid[i])
        scores.extend([i / len(sid) for j in range(len(leaf))])
        leafs.extend(leaf)
    return leafs, scores

def check_has_answer(passage, answers):
    for answer in answers:
        if answer in passage:
            return True
    return False

def hard_neg_mining(candidates, filter_strs):
    hn_candidates = []
    for pid, candidate, score in candidates:
        passage = ' '.join(candidate)
        if check_has_answer(passage, filter_strs): continue
        hn_candidates.append([pid, score])
    return hn_candidates

semanticids = read_tsv(SEMANTICIDS_PATH)
psgs_list   = read_tsv(PSGS_PATH)

semantic_tree = build_semantic_tree(semanticids)
pid2semanticid= dict([[pid, sid] for pid, sid in semanticids])
psgs_dict     = dict([[psgs[0], [psgs[1], psgs[2]]] for psgs in psgs_list])


nq_train_small_datas = read_json(NQ_TRAIN_SMALL_PATH)

for data in nq_train_small_datas:
    answers = data['answers']
    sid = data['semantic_id']
    negative_ctxs = data['negative_ctxs']
    
    leafs_pid, scores = same_sub_cluster(semantic_tree, sid)
    candidates = [[leaf_pid, psgs_dict[leaf_pid], score] for leaf_pid, score in zip(leafs_pid, scores)]
    print('{} -> '.format(len(candidates)), end='')
    candidates = hard_neg_mining(candidates, answers)
    print(len(candidates))
    assert len(candidates) > 0

    semantic_negative_ctxs = []
    for pid, score in candidates:
        title = psgs_dict[pid][1].replace('"', '')
        text  = psgs_dict[pid][0]
        if text[0] == '"' and text[-1] == '"':
            text = text[1:-1]

        score = score
        title_score = 0
        passage_id = pid
        semantic_id= pid2semanticid[pid]
        semantic_negative_ctxs.append({
            'title': title,
            'text': text,
            'score': score,
            'title_score': title_score,
            'passage_id': pid,
            'semantic_id': semantic_id,
            'type': 'semantic hard negative'
        })
    data['negative_ctxs'] = semantic_negative_ctxs


with open('./tmp/nq-train-small-shardneg.json', 'w', encoding='utf-8') as fr:
    json.dump(nq_train_small_datas, fr, indent=4)
    