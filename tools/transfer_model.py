import torch

MODEL_1_PATH = '/share/nas165/amian/experiments/nlp/DPR/outputs/2022-07-15/15-28-28/output/dpr_biencoder.38'
MODEL_2_PATH = './assets/transformers/t5-base/pytorch_model.bin'

OUTPUT_PATH  = './assets/transformers/st5-base-encoder-pretrained/pytorch_model.bin'


def show_model(model, filter=None):
    for key in model:
        # filter = key if filter == None else filter
        # if filter not in key: continue
        print('key: {}, item: {}'.format(key, type(model[key])))
        
        # try:
        #     for k in model[key]:
        #         filter = k if filter == None else filter
        # except:
        #     ...

def show_diff_model(model_A, model_B):
    for key in model_A:
        prefix = 'O' if key in model_B else 'X'
        print('({}) {}'.format(prefix, key))
        
        # try:
        #     for k in model_A[key]:
        #         prefix = 'O' if k in model_B[key] else 'X'
        #         print('({}) {}'.format(prefix, k))
        # except:
        #     ...

def combine_model(model_A, model_B, filter=None):
    for key in model_A:
        # filter = key if filter == None else filter
        # if filter in key and key in model_B:
        try:
            model_A[key] = model_B[key]
            print('replaced: [{}]'.format(key))
        except:
            ...
        # try:
        #     for k in model_A[key]:
        #         filter = k if filter == None else filter
        #         if filter in k and k in model_B[key]:
        #             model_A[key][k] = model_B[key][k]
        #             print('replace: [{}]'.format(k))
        # except:
        #     ...
    return model_A


device = torch.device('cpu')
model_1 = torch.load(MODEL_1_PATH, map_location=device) 
model_2 = torch.load(MODEL_2_PATH, map_location=device) 

# print(type(model_2))
# print(model_1['model_dict'].keys())

n_model_1 = {}
for key in model_1['model_dict']:
    if 'question_model' in key:
        nkey = key.replace('question_model.', '')
        n_model_1[nkey] = model_1['model_dict'][key]

model_1 = n_model_1

# show_model(model_1['model_dict'], filter='question_model')
# show_model(model_1)
# show_model(model_2)
show_diff_model(model_2, model_1)
replaced_model = combine_model(model_2, model_1)
torch.save(replaced_model, OUTPUT_PATH)