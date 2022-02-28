# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from src import utils


class Embeddings(nn.Module):

    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) / math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class MyTransformerDecoder(nn.Module):

    def __init__(self,
                 len_vocab,
                 d_model,
                 nhead,
                 nlayer,
                 tie_readout=True,
                 dim_ff=2048,
                 dropout=0.1,
                 activation='relu'):
        super(MyTransformerDecoder, self).__init__()
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout,
                                               activation)
        dec_norm = nn.LayerNorm(d_model)
        dec_layer.apply(utils.weights_init)

        self.d_model = d_model
        self.tgt_emb = nn.Sequential(Embeddings(len_vocab, d_model),
                                     PositionalEncoding(d_model))
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, nlayer, dec_norm)
        self.out_proj = nn.Linear(d_model, len_vocab, bias=False)

        if tie_readout:
            self.out_proj.weight = self.tgt_emb[0].lut.weight
        else:
            utils.weights_init(self.out_proj)

    def forward(self, tgt, tgt_mask, tgt_key_padding_mask, memory,
                memory_key_padding_mask):
        tgt_vec = self.drop(self.tgt_emb(tgt))
        output_sense = self.decoder(
            tgt_vec.permute(1, 0, 2),
            memory.permute(1, 0, 2),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask).permute(1, 0, 2)
        # output_sense = torch.log_softmax(self.out_proj(output_sense), dim=-1)
        return self.out_proj(output_sense)


class MyModel(nn.Module):

    def __init__(self,
                 len_vocab,
                 dec_d_model,
                 dec_nhead,
                 dec_nlayer,
                 dec_d_ff=1024,
                 tie_readout=True,
                 dec_dropout=0.2,
                 dec_activation='relu'):
        super(MyModel, self).__init__()
        self.dec_model = MyTransformerDecoder(len_vocab,
                                              dec_d_model,
                                              dec_nhead,
                                              dec_nlayer,
                                              tie_readout=tie_readout,
                                              dim_ff=dec_d_ff,
                                              dropout=dec_dropout,
                                              activation=dec_activation)
        self.dropout = nn.Dropout(dec_dropout)

    def forward(self, input_ids, attention_mask, memory, memory_mask):
        subsequent_mask = utils.generate_square_subsequent_mask(
            input_ids.size(1)).to(input_ids.device)
        tgt_key_padding_mask = (attention_mask == 1)
        memory_key_padding_mask = (memory_mask == 1)
        output = self.dec_model(input_ids, subsequent_mask,
                                tgt_key_padding_mask, memory,
                                memory_key_padding_mask)
        return output


if __name__ == "__main__":
    from dataset import MyDataset
    from torch.utils import data
    dataset_path = '/home/kcl/workspace/codwoe/data/dev/en.dev.json'
    tok_path = '/home/kcl/workspace/codwoe/data/tokenizer/en.json'
    dataset = MyDataset(dataset_path, tok_path)
    dataloader = data.DataLoader(dataset,
                                 batch_size=200,
                                 collate_fn=dataset.padding_collate)
    model = MyModel(dataset.tokenizer.get_vocab_size(), 256, 8, 6)
    for batch in dataloader:
        output = model(batch)
        print(output.shape)
