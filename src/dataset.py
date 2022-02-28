# -*- coding: utf-8 -*-
import torch
import json
from torch.utils import data
from torch.utils.data._utils.collate import default_collate


class MyDataset(data.Dataset):

    def __init__(self,
                 dataset_path,
                 tokenizer,
                 sgns=True,
                 char=True,
                 electra=True):
        self.sgns = sgns
        self.char = char
        self.electra = electra
        self.tokenizer = tokenizer
        self.dataset = self.__load_dataset(dataset_path)

    def __load_dataset(self, dataset_path):
        dataset = []
        ori_data = json.load(open(dataset_path))
        for item in ori_data:
            new_item = {'id': item['id']}
            if 'gloss' in item:
                gloss = item['gloss']
                gloss_ids = self.tokenizer.encode(gloss).ids
                gloss_tensor = torch.LongTensor(gloss_ids)
                new_item['gloss'] = gloss
                new_item['gloss_ids'] = gloss_tensor[:-1]
                new_item['gloss_gold'] = gloss_tensor[1:]
                new_item['gloss_attn_mask'] = torch.ByteTensor(
                    [0] * (gloss_tensor.size(0) - 1))
            else:
                new_item['gloss'] = ''
                gloss_tensor = torch.LongTensor(
                    [self.tokenizer.token_to_id('<s>')])
                new_item['gloss_ids'] = gloss_tensor
                new_item['gloss_gold'] = gloss_tensor
                new_item['gloss_attn_mask'] = torch.ByteTensor(
                    [0] * (gloss_tensor.size(0)))

            mem_mask = []
            memory = []
            if self.sgns:
                memory.append(item['sgns'])
                # new_item['sgns'] = torch.FloatTensor(item['sgns'])
                mem_mask.append(0)
            if self.char:
                memory.append(item['char'])
                # new_item['char'] = torch.FloatTensor(item['char'])
                mem_mask.append(0)
            if self.electra:
                memory.append(item['electra'])
                # new_item['electra'] = torch.FloatTensor(item['electra'])
                mem_mask.append(0)
            new_item['mem_mask'] = torch.ByteTensor(mem_mask)
            new_item['memory'] = torch.FloatTensor(memory)
            dataset.append(new_item)
        return dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def padding(self, tokens, max_len, pad_token_id):
        pad_ids = torch.LongTensor([pad_token_id] * (max_len - tokens.size(0)))
        tokens = torch.cat([tokens, pad_ids], dim=0)
        return tokens

    def padding_collate(self, list_of_items):
        max_len = max([x['gloss_ids'].size(0) for x in list_of_items])
        for item in list_of_items:
            item['gloss_ids'] = self.padding(item['gloss_ids'], max_len, 0)
            item['gloss_gold'] = self.padding(item['gloss_gold'], max_len, 0)
            item['gloss_attn_mask'] = self.padding(item['gloss_attn_mask'],
                                                   max_len, 1)
        return default_collate(list_of_items)


if __name__ == '__main__':
    from tokenizers import Tokenizer
    dataset_path = '/home/kcl/workspace/codwoe/data/dev/en.dev.json'
    tok_path = '/home/kcl/workspace/codwoe/data/tokenizer/en.json'
    dataset = MyDataset(dataset_path, Tokenizer.from_file(tok_path))
    dataloader = data.DataLoader(dataset,
                                 batch_size=256,
                                 num_workers=4,
                                 collate_fn=dataset.padding_collate)
    for batch in dataloader:
        import pdb
        pdb.set_trace()
        print(batch['gloss_ids'].shape)
        print(batch['gloss_attn_mask'].shape)
