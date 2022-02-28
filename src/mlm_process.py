# -*- coding: utf-8 -*-
import math
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F

PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 4
SPECIAL_TOKEN_IDS = [0, 1, 2, 3, 4]


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


def mlm_process(input_ids,
                num_tokens,
                mask_prob=0.15,
                random_token_prob=0.2,
                replace_prob=0.8):
    no_mask = mask_with_tokens(input_ids, SPECIAL_TOKEN_IDS)
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)

    # get mask indices
    # mask_indices = torch.nonzero(mask, as_tuple=True)

    # mask input with mask tokens with probability of
    # `replace_prob` (keep tokens the same with probability 1 - replace_prob)
    masked_input = input_ids.clone().detach()
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        assert num_tokens is not None, \
            'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
        random_token_prob = prob_mask_like(input_ids, random_token_prob)
        random_tokens = torch.randint(0,
                                      num_tokens,
                                      input_ids.shape,
                                      device=input_ids.device)
        # random_no_mask = mask_with_tokens(random_tokens, SPECIAL_TOKEN_IDS)
        # random_token_prob &= ~random_no_mask
        random_token_prob &= ~no_mask
        random_indices = torch.nonzero(random_token_prob, as_tuple=True)
        masked_input[random_indices] = random_tokens[random_indices]
    # [mask] input
    replace_prob = prob_mask_like(input_ids, replace_prob)
    masked_input = masked_input.masked_fill(mask * replace_prob, MASK_TOKEN_ID)
    # mask out any tokens to padding tokens that were not originally going to be masked
    labels = input_ids.masked_fill(~mask, PAD_TOKEN_ID)
    return masked_input, labels
