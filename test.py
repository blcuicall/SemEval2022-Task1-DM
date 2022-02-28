# -*- coding: utf-8 -*-
import sys
import os
import math
import glob
import json
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from tokenizers import Tokenizer
from copy import deepcopy

from src import dataset, model, utils, options
import evaluate


def load_checkpoint(mymodel, path):
    checkpoint = torch.load(path)
    mymodel.load_state_dict(checkpoint)
    return mymodel


def test(args, test_iter, tokenizer):
    mymodel = model.MyModel(tokenizer.get_vocab_size(), args.dec_dmodel,
                            args.dec_nhead, args.dec_nlayer, args.dec_dff,
                            args.dec_tie_readout)

    model_list = []
    for path in args.restore:
        cur_model = load_checkpoint(deepcopy(mymodel), path).cuda()
        cur_model.eval()
        model_list.append(cur_model)
    del mymodel

    test_hyp = evaluate.beam_search(test_iter, tokenizer, model_list)

    test_result = []
    for idx, sense in enumerate(test_hyp):
        test_result.append({
            'id': test_iter.dataset[idx]['id'],
            'gloss': sense
        })

    with open(args.result_path, 'w') as fw:
        fw.write(json.dumps(test_result, ensure_ascii=False))
    return 1


def main(args):
    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    test_data = dataset.MyDataset(args.test_path,
                                  tokenizer,
                                  sgns=args.sgns,
                                  char=args.char,
                                  electra=args.electra)
    test_iter = data.DataLoader(test_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=test_data.padding_collate,
                                pin_memory=True)
    print(args)

    if test(args, test_iter, tokenizer):
        print("Inference Done.")


if __name__ == '__main__':
    args = options.eval_options()
    sys.exit(main(args))
