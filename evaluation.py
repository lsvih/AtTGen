from typing import Dict, List

import torch
from tqdm import tqdm


def evaluate(model, val_loader, config):
    model.eval()
    f1_triple = F1Triplet(config)
    print('Evaluating...')
    with torch.no_grad():
        for batch_ndx, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
            output = model(sample, do_train=False)
            f1_triple(output["decode_result"], output["spo_gold"])
    result = f1_triple.get_metric()
    return result["fscore"]


class F1Triplet:
    def __init__(self, config):
        self.skip_subject = config.skip_subject
        if self.skip_subject:
            self.get_seq = lambda dic: (dic["object"], dic["predicate"])
        else:
            self.get_seq = lambda dic: (dic["subject"], dic["object"], dic["predicate"])
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()
        f1, p, r = 2 * self.A / (self.B + self.C), self.A / self.B, self.A / self.C
        result = {"precision": p, "recall": r, "fscore": f1}
        return result

    def __call__(self, predictions: List[List[Dict[str, str]]], gold_labels: List[List[Dict[str, str]]]):
        for g, p in zip(gold_labels, predictions):
            g_set = set("_".join(self.get_seq(gg)) for gg in g)
            p_set = set("_".join(self.get_seq(pp)) for pp in p)
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)


def load_model(_model, path):
    _model.load_state_dict(torch.load(path, map_location="cpu"))
    return _model


if __name__ == '__main__':
    from model import AtTGenModel as Model
    from dataloader import TreeDataset, collate
    from torch.utils.data import DataLoader
    import args

    config = args.get_args()
    config.name = 'ship'
    config.batch_size = 5
    # config.tokenizer = 'base'
    # config.data_dir = './data/nyt'
    # config.ontology_vocab = 'relation_vocab.json'

    val_dataset = TreeDataset(data_dir=config.data_dir, data_type='train', ontology_vocab=config.ontology_vocab, tokenizer=config.tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1,
                            collate_fn=collate, pin_memory=True)

    model = Model(config)
    model = load_model(model, './runs/{}_best'.format(config.name))
    model.to(config.device)
    print(evaluate(model, val_loader, config))
