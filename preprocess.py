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


def process_CNShipNet(raw_path, data_path, subject_guide=False):
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
    if subject_guide:
        word_vocab.add('[subject]')
    attribute_vocab = {attr: i for i, attr in enumerate(attribute_vocab)}
    word_vocab = {word: i for i, word in enumerate(word_vocab)}
    open(os.path.join(target_data_path, 'attribute_vocab.json'), 'w').write(
        json.dumps(attribute_vocab, ensure_ascii=False))
    open(os.path.join(target_data_path, 'word_vocab.json'), 'w').write(json.dumps(word_vocab, ensure_ascii=False))
    # build data
    for file in file_map.keys():
        data = json.loads(open(os.path.join(raw_data_path, file)).read())
        f = open(os.path.join(target_data_path, file_map[file]), 'w')
        for instance in data:
            spo_list = instance['relation_list']
            text = instance['text']
            if subject_guide:
                assert len(set(map(lambda x: x['subject'], spo_list))) == 1
                text = "[sub]{}[/sub]{}".format(spo_list[0]['subject'], text)
            tokens, raw_tokens = tokenizer(text)
            f.write(json.dumps({
                'text_id': instance['id'],
                'text': text,
                'tokens': tokens,
                'raw_tokens': raw_tokens,
                'spo_list': spo_list
            }, ensure_ascii=False) + '\n')
    print('Processing CNShipNet dataset done!')


def process_jave(raw_path, data_path):
    raw_data_path = os.path.join(raw_path, 'jave')
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError('Raw data path not found!')
    target_data_path = os.path.join(data_path, 'jave')
    if not os.path.exists(target_data_path):
        os.makedirs(target_data_path)
    tokenizer = load_tokenizer('char')
    file_map = {
        'jdair.jave.train.txt': 'train_data.json',
        'jdair.jave.valid.txt': 'validate_data.json',
        'jdair.jave.test.txt': 'test_data.json'
    }
    attribute_vocab = open(os.path.join(raw_data_path, 'attribute_vocab.txt')).read().strip().splitlines()
    word_vocab = open(os.path.join(raw_data_path, 'word_vocab.txt')).read().strip().splitlines()
    attribute_vocab = {'@' + attr: i for i, attr in enumerate(attribute_vocab)}
    word_vocab = {word: i for i, word in enumerate(word_vocab)}
    word_vocab['#'] = len(word_vocab)
    open(os.path.join(target_data_path, 'attribute_vocab.json'), 'w').write(
        json.dumps(attribute_vocab, ensure_ascii=False))
    open(os.path.join(target_data_path, 'word_vocab.json'), 'w').write(json.dumps(word_vocab, ensure_ascii=False))

    for file in file_map.keys():
        data = open(os.path.join(raw_data_path, file)).read().strip().splitlines()
        f = open(os.path.join(target_data_path, file_map[file]), 'w')
        for line in data:
            instance = line.split('\t')
            spo_list = []
            # process tags
            doc_p = instance[3].strip().lower()
            index = 0
            while index < len(doc_p):
                if doc_p[index] == "<":
                    index += 1
                    attr = ""
                    value = ""
                    while doc_p[index] != ">":
                        attr += doc_p[index]
                        index += 1
                    index += 1
                    while doc_p[index] != "<":
                        value += doc_p[index]
                        index += 1
                    index += 1
                    assert doc_p[index] == "/"
                    index += 1
                    while doc_p[index] != ">":
                        index += 1
                    index += 1
                    spo_list.append({
                        'subject': '#',
                        'predicate': '@' + attr,
                        'object': value
                    })
                else:
                    index += 1
            text = instance[2].lower()
            text = '#' + text
            tokens, raw_tokens = tokenizer(text)
            f.write(json.dumps({
                'text_id': instance[0],
                'text': text,
                'tokens': tokens,
                'raw_tokens': raw_tokens,
                'spo_list': spo_list
            }, ensure_ascii=False) + '\n')
    print('Processing JAVE dataset done!')


def main(config):
    # Processing dataset
    if config.dataset == 'CNShipNet':
        process_CNShipNet(config.raw_data_dir, config.data_dir, config.subject_guide)
    if config.dataset == 'jave':
        process_jave(config.raw_data_dir, config.data_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='CNShipNet', help='Dataset name')
    args.add_argument('--raw_data_dir', type=str, default='./raw_data', help='Raw data directory')
    args.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    args.add_argument('--subject_guide', type=bool, default=False, help='Whether to use subject guide')
    args = args.parse_args()
    main(args)
