import argparse

import torch

from utils import get_device


def get_args():
    parser = argparse.ArgumentParser(description='configuration')

    parser.add_argument("--name",
                        default="1",
                        type=str,
                        help="Experiment name, for logging and saving models")
    parser.add_argument("--do_train",
                        action='store_true',
                        default=False,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        default=False,
                        help="Whether to run eval on the test set.")
    parser.add_argument("--data_dir",
                        default="./data/jave",
                        type=str,
                        help="The input data dir.")
    parser.add_argument("--word_vocab",
                        default="word_vocab.json",
                        type=str,
                        help="The vocabulary file.")
    parser.add_argument("--ontology_vocab",
                        default="attribute_vocab.json",
                        type=str,
                        help="The ontology class file.")
    parser.add_argument("--tokenizer", default="char",
                        type=str,
                        help="The tokenizer type.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="The random seed for initialization")
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='2',
                        help="The GPU ids")

    # Hyperparameters
    # Batch size
    parser.add_argument("--batch_size",
                        default=512,
                        type=int,
                        help="Total batch size for training.")
    # Learning rate
    parser.add_argument("--lr",
                        default=2e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    # Epochs
    parser.add_argument("--epoch",
                        default=40,
                        type=int,
                        help="Total number of training epochs to perform.")
    # emb_dim
    parser.add_argument("--emb_dim",
                        default=200,
                        type=int,
                        help="The dimension of the embedding")
    # encode_dim
    parser.add_argument("--encode_dim",
                        default=200,
                        type=int,
                        help="The dimension of the encoding")
    # skip subject
    parser.add_argument("--skip_subject",
                        default=True,
                        type=bool,
                        help="Whether to skip the subject")

    args = parser.parse_args()
    if args.gpu_ids == "":
        n_gpu = 0
        device = torch.device('cpu')
    else:
        gpu_ids = [int(device_id) for device_id in args.gpu_ids.split()]
        args.gpu_ids = gpu_ids
        device, n_gpu = get_device(gpu_ids[0])
        if n_gpu > 1:
            n_gpu = len(gpu_ids)
    args.device = device
    args.n_gpu = n_gpu

    return args
