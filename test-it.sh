#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
LANG=it
DATA_DIR=../data

python -u test.py \
    --test-path ${DATA_DIR}/test/${LANG}.test.defmod.json \
    --tokenizer-path ${DATA_DIR}/tokenizer/${LANG}.json \
    --sgns --char \
    --dec-dmodel 256 \
    --dec-nhead 8 \
    --dec-nlayer 3 \
    --dec-dff 1024 \
    --dec-tie-readout \
    --batch-size 256 \
    --restore checkpoints/${LANG}-seed-1111/model-best.pt \
    --restore checkpoints/${LANG}-seed-2222/model-best.pt \
    --restore checkpoints/${LANG}-seed-3333/model-best.pt \
    --restore checkpoints/${LANG}-seed-4444/model-best.pt \
    --restore checkpoints/${LANG}-seed-5555/model-best.pt \
    --result-path results/test.${LANG}.json \
    --seed 42 \
    2>&1

    #--test-path ${DATA_DIR}/test/${LANG}.test.defmod.json \
