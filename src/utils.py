# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel


def obtain_tokenizer(model_arch, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_arch,
        cache_dir=cache_dir,
        proxies={"https": "202.112.194.62:7654"})
    return tokenizer


def obtain_config(model_arch, cache_dir):
    config = AutoConfig.from_pretrained(
        model_arch,
        cache_dir=cache_dir,
        proxies={"https": "202.112.194.62:7654"})
    return config


def obtain_pretrained_encoder(model_arch, cache_dir):
    encoder = AutoModel.from_pretrained(
        model_arch,
        cache_dir=cache_dir,
        proxies={"https": "202.112.194.62:7654"})
    return encoder


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.ones_(m.bias)
    if isinstance(m, nn.LSTMCell):
        nn.init.xavier_uniform_(m.weight_hh)
        nn.init.xavier_uniform_(m.weight_ih)
        nn.init.ones_(m.bias_hh)
        nn.init.ones_(m.bias_ih)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
    # mask == 1, float(0.0))
    return ~mask


def to_cuda(item):
    for k in item:
        if isinstance(item[k], dict):
            to_cuda(item[k])
        elif isinstance(item[k], torch.Tensor):
            item[k] = item[k].cuda()
    return item


class Vocab(object):

    def __init__(self, vocab_path):
        self.itos = self.read_vocab(vocab_path)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def read_vocab(self, vocab_path):
        itos = []
        with open(vocab_path) as fr:
            for line in fr:
                word = line.strip().split('\t')[0]
                itos.append(word)
        return itos


class LabelSmoothKLDiv(nn.Module):
    "Implement label smoothing."

    def __init__(self,
                 size,
                 ignore_idx=-100,
                 smoothing=0.1,
                 reduction='batchmean'):
        super(LabelSmoothKLDiv, self).__init__()
        self.ignore_idx = ignore_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.criterion = nn.KLDivLoss(reduction=reduction)

    def forward(self, src, trg):
        assert src.size(1) == self.size
        with torch.no_grad():
            true_dist = src.clone()
            true_dist.fill_(self.smoothing / (self.size - 2))
            true_dist.scatter_(1, trg.unsqueeze(1), self.confidence)
            true_dist[:, self.ignore_idx] = 0
            mask = torch.nonzero(trg == self.ignore_idx)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(src, true_dist)


class InverseSqureRootOptim(object):
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, init_lr, lr, min_lr, warmup):
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0
        self.warmup = warmup
        self.init_lr = init_lr
        self.lr = lr
        self.min_lr = min_lr
        self.lr_step = (lr - init_lr) / warmup
        self.decay_factor = lr * warmup**0.5

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if step < self.warmup:
            lr = self.init_lr + step * self.lr_step
        else:
            lr = max(self.decay_factor * step**-0.5, self.min_lr)
        return lr
