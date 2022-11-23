import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import args
from dataloader import TreeDataset, collate
from evaluation import evaluate, load_model
from model import AtTGenModel
from train import train

cudnn.benchmark = True
cudnn.deterministic = False
cudnn.enabled = True


def main(config):
    if config.do_train and not config.do_eval:
        train_dataset = TreeDataset(data_dir=config.data_dir, data_type='train', word_vocab=config.word_vocab,
                                    ontology_vocab=config.ontology_vocab, tokenizer=config.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
                                  collate_fn=collate, pin_memory=True)
        val_dataset = TreeDataset(data_dir=config.data_dir, data_type='validate', word_vocab=config.word_vocab,
                                  ontology_vocab=config.ontology_vocab, tokenizer=config.tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4,
                                collate_fn=collate, pin_memory=True)
        model = AtTGenModel(config)
        if config.n_gpu > 1:
            print('Using {} GPUs'.format(config.n_gpu))
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model.to(config.device)
        train(model, train_loader, val_loader, config)
        test_dataset = TreeDataset(data_dir=config.data_dir, data_type='test', word_vocab=config.word_vocab,
                                   ontology_vocab=config.ontology_vocab, tokenizer=config.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4,
                                 collate_fn=collate, pin_memory=True)
        # Load Best Model
        model = load_model(model, './runs/{}_best'.format(config.name))
        model.to(config.device)
        score = evaluate(model, test_loader, config)
        print("Test F1 score: {}".format(score))
    if config.do_eval:
        model = AtTGenModel(config)
        test_dataset = TreeDataset(data_dir=config.data_dir, data_type='test', word_vocab=config.word_vocab,
                                   ontology_vocab=config.ontology_vocab, tokenizer=config.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4,
                                 collate_fn=collate, pin_memory=True)
        # Load Best Model
        model = load_model(model, './runs/{}_best'.format(config.name))
        model.to(config.device)
        score = evaluate(model, test_loader, config)
        print("Test F1 score: {}".format(score))


if __name__ == '__main__':
    args = args.get_args()
    # Fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    main(args)
