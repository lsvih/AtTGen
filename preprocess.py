import argparse
import json
import os

from tokenizer import load_tokenizer


def process_en_data(raw_path, data_path):
    raw_data_path = os.path.join(raw_path, 'nyt')
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError('Raw data path not found!')
    target_data_path = os.path.join(data_path, 'nyt')
    if not os.path.exists(target_data_path):
        os.makedirs(target_data_path)
    tokenizer = load_tokenizer('base')
    pass


def process_CNShipNet(raw_path, data_path):
    raw_data_path = os.path.join(raw_path, 'CNShipNet')
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError('Raw data path not found!')
    target_data_path = os.path.join(data_path, 'CNShipNet')
    if not os.path.exists(target_data_path):
        os.makedirs(target_data_path)
    tokenizer = load_tokenizer('chn')
    file_map = {
        'train.json': 'train_data.json',
        'dev.json': 'validate_data.json',
        'test.json': 'test_data.json'
    }
    attribute_vocab = set()
    word_vocab = set()
    # build word_vocab and attribute_vocab and tokenizer word set
    for file in file_map.keys():
        data = json.loads(open(os.path.join(raw_data_path, file)).read())
        for instance in data:
            for spo in instance['relation_list']:
                if spo['predicate'].startswith('@'):
                    attribute_vocab.add(spo['predicate'])
            for token in tokenizer(instance['text'])[0]:
                word_vocab.add(token)
    attriute_vocab = {attr: i for i, attr in enumerate(attribute_vocab)}
    word_vocab = {word: i for i, word in enumerate(word_vocab)}
    open(os.path.join(target_data_path, 'attribute_vocab.json'), 'w').write(
        json.dumps(attriute_vocab, ensure_ascii=False))
    open(os.path.join(target_data_path, 'word_vocab.json'), 'w').write(json.dumps(word_vocab, ensure_ascii=False))
    # build data
    for file in file_map.keys():
        data = json.loads(open(os.path.join(raw_data_path, file)).read())
        f = open(os.path.join(target_data_path, file_map[file]), 'w')
        for instance in data:
            text = instance['text']
            tokens, raw_tokens = tokenizer(text)
            spo_list = instance['relation_list']
            f.write(json.dumps({
                'text_id': instance['id'],
                'text': text,
                'tokens': tokens,
                'raw_tokens': raw_tokens,
                'spo_list': spo_list
            }, ensure_ascii=False) + '\n')
    print('Processing CNShipNet dataset done!')


def main(config):
    # Processing dataset
    if config.dataset == 'CNShipNet':
        process_CNShipNet(config.raw_data_dir, config.data_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='CNShipNet', help='Dataset name')
    args.add_argument('--raw_data_dir', type=str, default='./raw_data', help='Raw data directory')
    args.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    args = args.parse_args()
    main(args)
