TRAIN_DATASET=aishell_masked_train
DEV_DATASET=aishell_masked_dev

CONFIG=biencoder_aishell
ENCODER=hf_bert_entity_chinese

CUDA_VISIBLE_DEVICES=0,1 python3 train_dense_encoder.py \
    train_datasets=[$TRAIN_DATASET]                     \
    dev_datasets=[$DEV_DATASET]                         \
    train=$CONFIG                                       \
    encoder=$ENCODER                                    \
    output_dir=output