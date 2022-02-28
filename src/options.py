# -*- coding: utf-8 -*-
import argparse


def add_dataset_args(parser):
    parser.add_argument("--train-path", type=str)
    parser.add_argument("--dev-path", type=str)
    parser.add_argument("--test-path", type=str)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--sgns", action='store_true')
    parser.add_argument("--char", action='store_true')
    parser.add_argument("--electra", action='store_true')


def add_model_args(parser):
    parser.add_argument("--dec-dmodel", type=int, default=256)
    parser.add_argument("--dec-nhead", type=int, default=8)
    parser.add_argument("--dec-nlayer", type=int, default=6)
    parser.add_argument("--dec-dff", type=int, default=1024)
    parser.add_argument("--dec-tie-readout", action='store_true')
    parser.add_argument("--dec-activation", type=str, default='relu')


def add_train_args(parser):
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save", type=str)
    parser.add_argument("--init-lr", type=float, default=1e-7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-9)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--clip-norm", type=float, default=0.1)
    parser.add_argument("--dec-dropout", type=float, default=0.3)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--mlm-task", action='store_true')


def add_eval_args(parser):
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--restore", type=str, action='append')
    parser.add_argument("--result-path", type=str)


def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    add_dataset_args(parser)
    add_model_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    return args


def eval_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    add_dataset_args(parser)
    add_model_args(parser)
    add_eval_args(parser)
    args = parser.parse_args()
    return args
