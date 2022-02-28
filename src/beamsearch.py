# -*- coding: utf-8 -*-
import torch
import time


class Beam(object):

    def __init__(self, beam_size, bos_id, eos_id, cuda=False):
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.beam_scores = torch.zeros(beam_size)
        self.done = torch.ones(beam_size, dtype=torch.int)
        self.gen_seq = torch.LongTensor([bos_id] * beam_size).unsqueeze(1)
        self.seq_len = torch.zeros(beam_size)
        if cuda:
            self.__to_cuda()

    def __to_cuda(self):
        self.beam_scores = self.beam_scores.cuda()
        self.done = self.done.cuda()
        self.gen_seq = self.gen_seq.cuda()
        self.seq_len = self.seq_len.cuda()

    def __sort(self, word_prob):
        vocab_size = word_prob.size(1)
        if not torch.all(self.beam_scores == 0):
            old_scores = self.beam_scores[self.done == 1].unsqueeze(
                1).expand_as(word_prob)
            word_prob = word_prob + old_scores
        word_prob_flat = word_prob.reshape(-1)
        top_beam_idx = word_prob_flat.argsort(-1,
                                              True)[:self.done.sum().item()]
        scores = word_prob_flat[top_beam_idx]
        beam_id = top_beam_idx // vocab_size
        token_id = top_beam_idx % vocab_size

        beam_sort = torch.arange(start=self.beam_size, end=0,
                                 step=-1).to(beam_id.device)
        beam_sort[self.done == 0] = 0
        beam_sort = beam_sort.argsort(-1, True)
        beam_id = beam_sort.index_select(0, beam_id)

        new_beam_id = torch.arange(self.beam_size).to(beam_id.device)
        new_beam_id[self.done == 1] = beam_id
        new_token_id = torch.LongTensor([self.eos_id] * self.beam_size).to(
            token_id.device)
        new_token_id[self.done == 1] = token_id
        new_scores = self.beam_scores.index_select(-1, new_beam_id)
        new_scores[self.done == 1] = scores

        self.done[new_token_id == self.eos_id] = 0
        self.beam_scores = self.beam_scores.index_select(-1, new_beam_id)
        self.beam_scores[self.done == 1] = new_scores[self.done == 1]

        return new_beam_id, new_token_id

    def __penalty(self):
        self.beam_scores = self.beam_scores / self.seq_len

    def step(self, word_prob):
        if self.gen_seq.size(1) == 1:
            word_prob = word_prob[0].unsqueeze(0)
        beam_id, token_id = self.__sort(word_prob)

        if self.gen_seq.size(1) == 1:
            self.gen_seq = torch.cat(
                [self.gen_seq, token_id.unsqueeze(1)], dim=-1)
        else:
            self.gen_seq = self.gen_seq.index_select(0, beam_id)
            self.gen_seq = torch.cat(
                [self.gen_seq, token_id.unsqueeze(1)], dim=-1)
        self.seq_len[self.done == 1] += 1
        if torch.all(self.done == 0):
            self.__penalty()
        return self.gen_seq[self.done == 1], self.done.sum().item()

    def get_topn(self, topn):
        topn_idx = self.beam_scores.argsort(-1, True)[:topn]
        topn_scores = self.beam_scores[topn_idx]
        topn_seq = self.gen_seq[topn_idx]
        return topn_seq, topn_scores


class BeamBatch(object):

    def __init__(self, batch_size, beam_size, bos_id, eos_id, cuda=False):
        self.batch_size = batch_size
        self.active_beams = [beam_size] * batch_size
        self.beams = [
            Beam(beam_size, bos_id, eos_id, cuda) for _ in range(batch_size)
        ]

    def step(self, word_prob):
        word_prob_split = word_prob.split(self.active_beams, dim=0)
        gen_batch_seq = []
        for idx, (prob, beam) in enumerate(zip(word_prob_split, self.beams)):
            if prob.size(0) == 0:
                continue
            gen_seq, active_num = beam.step(prob)
            gen_batch_seq.append(gen_seq)
            self.active_beams[idx] = active_num

        gen_batch_seq = torch.cat(gen_batch_seq, dim=0)
        done = (max(self.active_beams) == 0)
        return gen_batch_seq, done

    def get_topn(self, topn):
        seqs = []
        scores = []
        for beam in self.beams:
            topn_seq, topn_scores = beam.get_topn(topn)
            seqs += topn_seq.tolist()
            scores += topn_scores.tolist()
        return seqs, scores

    def get_gen_seq(self):
        seqs = []
        for beam in self.beams:
            seqs.append(beam.gen_seq)
        seqs = torch.cat(seqs, dim=0)
        return seqs

    def expand_beams(self, origin):
        assert origin.size(0) == self.batch_size
        index = []
        for idx, num in enumerate(self.active_beams):
            cur_idx = [idx] * num
            index += cur_idx
        index = torch.LongTensor(index).to(origin.device)
        new = origin.index_select(0, index)
        return new
