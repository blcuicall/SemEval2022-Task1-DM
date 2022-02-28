# -*- coding: utf-8 -*-
import argparse
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--save-path", type=str)
    args = parser.parse_args()
    return args


def load_gloss(path):
    gloss = []
    dataset = json.load(open(path))
    for line in dataset:
        gloss.append(line['gloss'])
    return gloss


if __name__ == "__main__":
    args = options()
    train_gloss = load_gloss(args.input_file)
    tok = Tokenizer(BPE(unk_token="</unk>"))
    tok.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    tok.pre_tokenizer = Metaspace()
    tok.post_processor = TemplateProcessing(single="<s> $A </s>",
                                            pair="<s> $A </s> $B:1 </s>:1",
                                            special_tokens=[
                                                ("<s>", 1),
                                                ("</s>", 2),
                                            ])
    trainer = BpeTrainer(
        vocab_size=10000,
        special_tokens=["</pad>", "<s>", "</s>", "</unk>", "<mask>"])
    tok.train_from_iterator(train_gloss, trainer)
    tok.save(args.save_path)
