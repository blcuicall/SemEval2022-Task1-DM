# -*- coding: utf-8 -*-
import sys
import os
import time
import math
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils import data
from tokenizers import Tokenizer

from src import dataset, model, utils, options
from src.mlm_process import mlm_process
import evaluate


def save_checkpoint(mymodel, model_dir, epoch):
    save_path = os.path.join(model_dir, f'model-{epoch}.pt')
    torch.save(mymodel.state_dict(), save_path)


def load_checkpoint(model_with_loss, path):
    checkpoint = torch.load(path)
    if torch.cuda.device_count() > 1:
        model_with_loss.module.model.load_state_dict(checkpoint)
    else:
        model_with_loss.model.load_state_dict(checkpoint)


def train(args, train_iter, dev_iter, tokenizer):
    mymodel = model.MyModel(tokenizer.get_vocab_size(), args.dec_dmodel,
                            args.dec_nhead, args.dec_nlayer, args.dec_dff,
                            args.dec_tie_readout, args.dec_dropout,
                            args.dec_activation)
    criterion_ce = nn.NLLLoss(ignore_index=0, reduction='mean')
    criterion = utils.LabelSmoothKLDiv(tokenizer.get_vocab_size(),
                                       ignore_idx=0,
                                       smoothing=0.1,
                                       reduction='batchmean')

    base_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                   mymodel.parameters()),
                            betas=(0.9, 0.98),
                            eps=1e-9)
    advanced_optim = utils.InverseSqureRootOptim(base_optim, args.init_lr,
                                                 args.lr, args.min_lr,
                                                 args.warmup)
    mymodel = mymodel.cuda()

    sample_results_id = np.random.randint(0, 1000, (5, ))
    dev_tgt = [x['gloss'] for x in dev_iter.dataset]
    batch_num = len(train_iter)
    best_dev_loss = float('inf')
    best_score = 0
    no_improvement = 0
    for epoch in range(1, args.max_epoch + 1):
        total_loss_ce = 0
        total_loss = 0
        start_time = time.time()
        epoch_start_time = time.time()
        for i, item in enumerate(train_iter):
            mymodel.train()
            item = utils.to_cuda(item)
            logits = torch.log_softmax(mymodel(item['gloss_ids'],
                                               item['gloss_attn_mask'],
                                               item['memory'],
                                               item['mem_mask']),
                                       dim=-1)
            logits_flattern = logits.view(logits.size(0) * logits.size(1), -1)
            with torch.no_grad():
                loss_ce = criterion_ce(logits_flattern,
                                       item['gloss_gold'].view(-1))
            loss = criterion(logits_flattern, item['gloss_gold'].view(-1))

            if args.mlm_task:
                input_ids, labels = mlm_process(item['gloss_gold'],
                                                tokenizer.get_vocab_size())
                memory = torch.zeros_like(item['memory'])
                logits = torch.log_softmax(mymodel(input_ids,
                                                   item['gloss_attn_mask'],
                                                   memory, item['mem_mask']),
                                           dim=-1)
                logits_flattern = logits.view(
                    logits.size(0) * logits.size(1), -1)
                loss_mlm = criterion(logits_flattern, labels.view(-1))
                loss += loss_mlm

            total_loss_ce += loss_ce.item()
            total_loss += loss.item()
            loss = loss / args.update_interval
            loss.backward()

            if (i + 1) % args.update_interval == 0 or (i + 1) == batch_num:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, mymodel.parameters()),
                    args.clip_norm)
                advanced_optim.step()
                mymodel.zero_grad()

            if (i + 1) % args.log_interval == 0:
                elapsed = time.time() - start_time
                cur_loss_ce = total_loss_ce / args.log_interval
                cur_loss = total_loss / args.log_interval
                print("| Epoch {:3d} | {:d}/{:d} batches | {:.2f} ms/batch "
                      "| LR {:.9f} | Step {:d} | Loss {:.2f} | CELoss {:.2f} "
                      "| PPL {:.2f} ".format(
                          epoch, i + 1, batch_num,
                          elapsed * 1000 / args.log_interval,
                          advanced_optim._rate, advanced_optim._step, cur_loss,
                          cur_loss_ce, math.exp(cur_loss_ce)))
                total_loss_ce = 0
                total_loss = 0
                start_time = time.time()
        dev_loss, dev_loss_ce = evaluate.eval_loss(dev_iter, mymodel,
                                                   criterion_ce, criterion)
        # dev_hyp = evaluate.beam_search(dev_iter, tokenizer, mymodel,
        #                                sample_results_id, args.beam_size)
        dev_hyp = evaluate.greedy(dev_iter, tokenizer, mymodel,
                                  sample_results_id)
        dev_sent_bleu, dev_mover_score = evaluate.scores(dev_hyp, dev_tgt)
        cur_score = dev_sent_bleu + dev_mover_score

        if dev_loss < best_dev_loss:
            best_ckpt = True
            best_dev_loss = dev_loss
            best_score = cur_score
            no_improvement = 0
        elif cur_score > best_score:
            best_ckpt = True
            best_score = cur_score
            no_improvement = 0
        else:
            best_ckpt = False
            no_improvement += 1

        save_checkpoint(mymodel, args.save, epoch)
        if best_ckpt:
            save_checkpoint(mymodel, args.save, 'best')

        print('-' * 80)
        print("| End of epoch {:3d} | Time {:.2f}s | LR {:.9f} | Step {:d}".
              format(epoch,
                     time.time() - epoch_start_time, advanced_optim._rate,
                     advanced_optim._step))
        print(
            "| Loss {:.2f} | CELoss {:.2f} | PPL {:.2f} | Sent-BLEU {:.6f} | Mover {:.6f}"
            .format(dev_loss, dev_loss_ce, math.exp(dev_loss_ce),
                    dev_sent_bleu, dev_mover_score))
        print("| Not Improved {:d}".format(no_improvement))
        print('-' * 80)

        if no_improvement >= args.patience:
            break
    return 1


def main(args):
    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    train_data = dataset.MyDataset(args.train_path,
                                   tokenizer,
                                   sgns=args.sgns,
                                   char=args.char,
                                   electra=args.electra)
    dev_data = dataset.MyDataset(args.dev_path,
                                 tokenizer,
                                 sgns=args.sgns,
                                 char=args.char,
                                 electra=args.electra)
    train_iter = data.DataLoader(train_data,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=train_data.padding_collate,
                                 pin_memory=True)
    dev_iter = data.DataLoader(dev_data,
                               batch_size=args.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=dev_data.padding_collate,
                               pin_memory=True)
    print(args)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if train(args, train_iter, dev_iter, tokenizer):
        print("Done training.")


if __name__ == '__main__':
    args = options.train_options()
    sys.exit(main(args))
