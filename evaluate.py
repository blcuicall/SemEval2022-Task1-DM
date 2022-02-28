# -*- coding: utf-8 -*-
import os
import torch
import itertools
import collections
import time

import tqdm

os.environ["MOVERSCORE_MODEL"] = "distilbert-base-multilingual-cased"
import moverscore_v2 as mv_sc

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk import word_tokenize as tokenize

from src.beamsearch import BeamBatch
from src import utils


def eval_loss(data_iter, mymodel, criterion_ce, criterion):
    total_loss_ce = 0
    total_loss = 0
    mymodel.eval()
    with torch.no_grad():
        for item in data_iter:
            item = utils.to_cuda(item)
            logits = torch.log_softmax(mymodel(item['gloss_ids'],
                                               item['gloss_attn_mask'],
                                               item['memory'],
                                               item['mem_mask']),
                                       dim=-1)
            logits_flattern = logits.view(logits.size(0) * logits.size(1), -1)
            total_loss += criterion(logits_flattern,
                                    item['gloss_gold'].view(-1))
            total_loss_ce += criterion_ce(logits_flattern,
                                          item['gloss_gold'].view(-1))

    mean_loss = total_loss / len(data_iter)
    mean_loss_ce = total_loss_ce / len(data_iter)

    return mean_loss, mean_loss_ce


def greedy(data_iter, tokenizer, mymodel, sample_ids, max_len=50):
    mymodel.eval()
    results = []
    with torch.no_grad():
        for item in data_iter:
            item = utils.to_cuda(item)
            item['gloss_ids'] = item['gloss_ids'][:, 0].unsqueeze(1)
            item['gloss_attn_mask'] = torch.zeros_like(item['gloss_ids'])
            keep_decoding = [1] * item['gloss_ids'].size(0)
            decoded_words = [[] for _ in range(item['gloss_ids'].size(0))]
            for _ in range(max_len):
                output = torch.log_softmax(mymodel(item['gloss_ids'],
                                                   item['gloss_attn_mask'],
                                                   item['memory'],
                                                   item['mem_mask']),
                                           dim=-1)
                max_ids = output[:, -1].max(-1)[1]
                for k in range(len(keep_decoding)):
                    word_id = max_ids[k].item()
                    if keep_decoding[k]:
                        cur_word = tokenizer.id_to_token(word_id)
                        if cur_word != '</s>':
                            if cur_word != '<s>':
                                decoded_words[k].append(cur_word)
                        else:
                            keep_decoding[k] = 0
                item['gloss_ids'] = torch.cat(
                    [item['gloss_ids'],
                     max_ids.unsqueeze(1)], dim=1)
                item['gloss_attn_mask'] = torch.zeros_like(item['gloss_ids'])
                if max(keep_decoding) == 0:
                    break
            for k in range(len(decoded_words)):
                seq = ''.join(decoded_words[k]).replace('▁', ' ')
                results.append(seq)

    print("Decoded samples:")
    for idx in sample_ids:
        sample = results[idx]
        target = data_iter.dataset[idx]['gloss']
        print(f"Hypothesis: {sample}")
        print(f"Target: {target}")

    return results


def beam_search(data_iter,
                tokenizer,
                model_list,
                beam_size=5,
                topn=1,
                max_len=50):
    bos_id = tokenizer.token_to_id('<s>')
    eos_id = tokenizer.token_to_id('</s>')
    with torch.no_grad():
        results = []
        for item in tqdm.tqdm(data_iter):
            item = utils.to_cuda(item)
            beam_batch = BeamBatch(item['gloss_ids'].size(0),
                                   beam_size,
                                   bos_id,
                                   eos_id,
                                   cuda=True)
            item['gloss_ids'] = beam_batch.get_gen_seq()
            item['gloss_attn_mask'] = torch.zeros_like(item['gloss_ids'])

            ori_memory = item['memory']
            ori_inp_mask = item['mem_mask']
            item['memory'] = beam_batch.expand_beams(item['memory'])
            item['mem_mask'] = beam_batch.expand_beams(item['mem_mask'])

            for _ in tqdm.tqdm(range(max_len), leave=False):
                logits_sum = 0
                for model in model_list:
                    logits_sum += model(item['gloss_ids'],
                                        item['gloss_attn_mask'],
                                        item['memory'], item['mem_mask'])
                logits_sum /= len(model_list)
                output = torch.log_softmax(logits_sum, dim=-1)
                next_output_feed, done = beam_batch.step(output[:, -1])
                if done:
                    break
                item['gloss_ids'] = next_output_feed
                item['gloss_attn_mask'] = torch.zeros_like(next_output_feed)
                item['memory'] = beam_batch.expand_beams(ori_memory)
                item['mem_mask'] = beam_batch.expand_beams(ori_inp_mask)

            generated_seq, _ = beam_batch.get_topn(topn)
            for seq_id in generated_seq:
                seq = []
                for word_id in seq_id:
                    if word_id == bos_id:
                        continue
                    if word_id == eos_id:
                        break
                    seq.append(tokenizer.id_to_token(word_id))
                seq = ''.join(seq).replace('▁', ' ')
                # print(seq)
                results.append(seq)

        # print("Decoded samples: ")
        # for idx in sample_ids:
        #     sample = results[idx]
        #     target = data_iter.dataset[idx]['gloss']
        #     print(f"Hypothesis: {sample}")
        #     print(f"Target: {target}")
    return results


def bleu(pred, target, smoothing_function=SmoothingFunction().method4):
    return sentence_bleu([pred], target, smoothing_function=smoothing_function)


def mover_corpus_score(sys_stream, ref_streams, trace=0):
    """Adapted from the MoverScore github"""

    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]
    fhs = [sys_stream] + ref_streams
    corpus_score = 0
    pbar = tqdm.tqdm(desc="MvSc.", disable=None, total=len(sys_stream))
    for lines in itertools.zip_longest(*fhs):
        if None in lines:
            raise EOFError(
                "Source and reference streams have different lengths!")
        hypo, *refs = lines
        idf_dict_hyp = collections.defaultdict(lambda: 1.0)
        idf_dict_ref = collections.defaultdict(lambda: 1.0)
        corpus_score += mv_sc.word_mover_score(
            refs,
            [hypo],
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=False,
        )[0]
        pbar.update()
    pbar.close()
    corpus_score /= len(sys_stream)
    return corpus_score


def scores(pred_list, target_list):
    assert len(pred_list) == len(target_list)
    sent_bleu = 0
    for hyp, tgt in zip(pred_list, target_list):
        hyp = tokenize(hyp)
        tgt = tokenize(tgt)
        sent_bleu += bleu(hyp, tgt)
    sent_bleu /= len(pred_list)
    mover_score = mover_corpus_score(pred_list, [target_list])
    return sent_bleu, mover_score
