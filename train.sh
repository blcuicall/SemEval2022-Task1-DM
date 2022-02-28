#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
LANG=en
DATA_DIR=../data
MODEL_DIR=checkpoints/03-mlm-lr1e-3
mkdir -p $MODEL_DIR

python -u train.py \
    --train-path ${DATA_DIR}/train/${LANG}.train.json \
    --dev-path ${DATA_DIR}/dev/${LANG}.dev.json \
    --tokenizer-path ${DATA_DIR}/tokenizer/${LANG}.json \
    --sgns --char --electra \
    --dec-dmodel 256 \
    --dec-nhead 8 \
    --dec-nlayer 5 \
    --dec-dff 1024 \
    --dec-tie-readout \
    --dec-dropout 0.3\
    --batch-size 64 \
    --update-interval 4 \
    --save $MODEL_DIR \
    --init-lr 1e-7 \
    --lr 1e-3 --min-lr 1e-9 \
    --warmup 4000 \
    --clip-norm 0.1 \
    --log-interval 100 \
    --max-epoch 500 \
    --mlm-task \
    --seed 42 \
    2>&1 | tee $MODEL_DIR/training.log



