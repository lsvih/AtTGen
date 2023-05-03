from torch.utils.data import DataLoader

import args
from dataloader import TreeDataset, collate
from evaluation import evaluate, load_model, F1Triplet
from model import AtTGenModel
import torch
import json
from pptree import Node, print_tree


def run(config):
    config.data_dir = './data/jave_sample'
    model = AtTGenModel(config)
    test_dataset = TreeDataset(data_dir=config.data_dir, data_type='test', word_vocab=config.word_vocab,
                               ontology_vocab=config.ontology_vocab, tokenizer=config.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=collate, pin_memory=True)
    original_text = [json.loads(line)['text'] for line in open('./data/jave_sample/test_data.json').readlines()]

    print(len(test_loader), len(original_text))
    # Load Model
    model = load_model(model, './runs/jave_best')
    model.to('cpu')

    model.eval()
    f1_triple = F1Triplet(config)
    print('Evaluating...')
    with torch.no_grad():
        for batch_ndx, sample in enumerate(test_loader):
            output = model(sample, do_train=False)
            print('Original text: {}'.format(original_text[batch_ndx][1:]))
            print('Golden Standard: {}'.format(json.dumps(sample['spo_list'][0], ensure_ascii=False)))
            print('Extracted Attribute: {}'.format(json.dumps(output['decode_result'][0], ensure_ascii=False)))

            root_node = Node("ROOT")
            for i in range(len(output['decode_result'][0])):
                att_value = Node(output['decode_result'][0][i]["object"], root_node)
                att_name = Node(output['decode_result'][0][i]["predicate"], att_value)
            print_tree(root_node, horizontal=False)
            print('=========')
            f1_triple(output["decode_result"], output["spo_gold"])
    result = f1_triple.get_metric()

    print("Test F1 score: {}".format(result["fscore"]))


if __name__ == '__main__':
    args = args.get_args()
    run(args)
