import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tokenizer import load_tokenizer
from utils import find_entity_id_from_tokens, seq_padding, sort_all


class TreeDataset(Dataset):
    def __init__(self, data_dir: str = './data/jave', data_type: str = "train", tokenizer='char',
                 word_vocab: str = 'word_vocab.json', ontology_vocab: str = 'attribute_vocab.json',
                 order: List[str] = ("subject", "object", "predicate")):
        print('Loading {} data...'.format(data_type))
        self.data_dir = data_dir
        self.order = order
        self.word_vocab = json.load(open(os.path.join(data_dir, word_vocab)))
        self.ontology_vocab = json.load(open(os.path.join(data_dir, ontology_vocab)))
        vocab_size = len(self.word_vocab)
        ontology_class_token = {k: (v + vocab_size) for k, v in self.ontology_vocab.items()}
        self.word_vocab.update(ontology_class_token)
        if '[pre]' not in self.word_vocab:
            self.word_vocab['[pre]'] = len(self.word_vocab)
        if '<oov>' not in self.word_vocab:
            self.word_vocab['<oov>'] = len(self.word_vocab)
        self.ontology_class = list(ontology_class_token.keys())
        self.tokenizer = load_tokenizer(tokenizer)

        self.text = []
        self.text_length = []
        self.spo_list = []
        self.token_ids = []
        self.S1 = []
        self.S2 = []
        self.S_K1_in = []
        self.O_K1_in = []
        self.S_K2_in = []
        self.O_K2_in = []
        self.O1 = []
        self.O2 = []
        self.P1 = []
        self.P2 = []
        self.P_K1_in = []
        self.P_K2_in = []
        file = open(os.path.join(self.data_dir, "{}_data.json".format(data_type))).read().strip().split('\n')
        for line in tqdm(file):
            instance = json.loads(line)
            if data_type == 'train' or True:
                expanded_instances = self.spo_to_seq(instance["text"], instance["spo_list"], self.tokenizer,
                                                     self.ontology_class)
                instances = expanded_instances
            else:
                token = self.tokenizer(instance["text"])[0] + ['[pre]'] + self.ontology_class
                # instance['text'] = self.tokenizer.restore(token)
                instance['token'] = token
                instances = [instance]
            for instance in instances:
                text = instance["text"]
                spo_list = instance["spo_list"]
                text_id = []
                for c in instance['token']:
                    text_id.append(self.word_vocab.get(c, self.word_vocab["<oov>"]))
                if len(text_id) > 512:
                    continue
                else:
                    self.text_length.append(len(text_id))
                assert len(text_id) > 0
                self.token_ids.append(text_id)

                s_k1 = instance.get("s_k1", 0)
                s_k2 = instance.get("s_k2", 0)
                o_k1 = instance.get("o_k1", 0)
                o_k2 = instance.get("o_k2", 0)
                p_k1 = instance.get("p_k1", 0)
                p_k2 = instance.get("p_k2", 0)

                s1_gt = instance.get("s1_gt", [])
                s2_gt = instance.get("s2_gt", [])
                o1_gt = instance.get("o1_gt", [])
                o2_gt = instance.get("o2_gt", [])
                p1_gt = instance.get("p1_gt", [])
                p2_gt = instance.get("p2_gt", [])

                self.text.append(instance['token'])  # raw tokens
                self.spo_list.append(spo_list)

                self.S1.append(s1_gt)
                self.S2.append(s2_gt)
                self.O1.append(o1_gt)
                self.O2.append(o2_gt)
                self.P1.append(p1_gt)
                self.P2.append(p2_gt)
                self.S_K1_in.append([s_k1])
                self.S_K2_in.append([s_k2])
                self.O_K1_in.append([o_k1])
                self.O_K2_in.append([o_k2])
                self.P_K1_in.append([p_k1])
                self.P_K2_in.append([p_k2])

        self.token_ids = np.array(seq_padding(self.token_ids))

        # training
        self.S1 = np.array(seq_padding(self.S1))
        self.S2 = np.array(seq_padding(self.S2))
        self.O1 = np.array(seq_padding(self.O1))
        self.O2 = np.array(seq_padding(self.O2))
        self.P1 = np.array(seq_padding(self.P1))
        self.P2 = np.array(seq_padding(self.P2))

        # self.K1_in, self.K2_in = np.array(self.K1_in), np.array(self.K2_in)
        # only two time step are used for training
        self.S_K1_in = np.array(self.S_K1_in)
        self.S_K2_in = np.array(self.S_K2_in)
        self.O_K1_in = np.array(self.O_K1_in)
        self.O_K2_in = np.array(self.O_K2_in)
        self.P_K1_in = np.array(self.P_K1_in)
        self.P_K2_in = np.array(self.P_K2_in)

    def __getitem__(self, index):
        return (
            self.token_ids[index],
            self.S1[index],
            self.S2[index],
            self.O1[index],
            self.O2[index],
            self.P1[index],
            self.P2[index],
            self.S_K1_in[index],
            self.S_K2_in[index],
            self.O_K1_in[index],
            self.O_K2_in[index],
            self.P_K1_in[index],
            self.P_K2_in[index],
            self.text[index],  # original text
            self.text_length[index],  # token length
            self.spo_list[index],  # spo list
        )

    def __len__(self):
        return len(self.text)

    def spo_to_seq(self, text, spo_list, tokenizer, ontology_class):
        # The relative position of element in tree is calculated by raw_token
        tree = self.spo_to_tree(spo_list, self.order)
        tokens = tokenizer(text)[1] + ['[pre]'] + ontology_class  # raw_token
        _tokens = tokenizer(text)[0] + ['[pre]'] + ontology_class  # embellished token for attribute

        def to_ent(outp):
            ent1, ent2 = [[0] * len(tokens) for _ in range(2)]
            for name in outp:
                _id = find_entity_id_from_tokens(tokens, self.tokenizer(name)[1])
                ent1[_id] = 1
                ent2[_id + len(self.tokenizer(name)[1]) - 1] = 1
            return ent1, ent2

        def to_in_key(inp, name):
            # side effect!
            if not inp:
                return 0, 0

            k1 = find_entity_id_from_tokens(tokens, self.tokenizer(inp)[1])
            k2 = k1 + len(self.tokenizer(inp)[1]) - 1
            out = k1, k2
            return out

        results = []
        for t in tree:
            t1_in, t2_in, t1_out, t2_out, t3_out = t
            for name, ori_out, ori_in in zip(
                    self.order, (t1_out, t2_out, t3_out), (t1_in, t2_in, None)
            ):
                new_out = to_ent(ori_out)
                if name == "predicate":
                    p1, p2 = new_out
                    p_k1, p_k2 = to_in_key(ori_in, name)
                elif name == "subject":
                    s1, s2 = new_out
                    s_k1, s_k2 = to_in_key(ori_in, name)
                elif name == "object":
                    o1, o2 = new_out
                    o_k1, o_k2 = to_in_key(ori_in, name)
                else:
                    raise ValueError("should be in predicate, subject, object")

            result = {
                "text": tokenizer.restore(tokens),
                "token": _tokens,
                "raw_token": tokens,
                "spo_list": spo_list,
                "s_k1": s_k1,
                "s_k2": s_k2,
                "o_k1": o_k1,
                "o_k2": o_k2,
                "p_k1": p_k1,
                "p_k2": p_k2,
                "s1_gt": s1,
                "s2_gt": s2,
                "o1_gt": o1,
                "o2_gt": o2,
                "p1_gt": p1,
                "p2_gt": p2,
            }

            results.append(result)
        return results

    def spo_to_tree(self, spo_list: List[Dict[str, str]], order=("subject", "object", "predicate")):
        """return the ground truth of the tree: rel, subj, obj, used for teacher forcing.
        r: given text, one of the relations
        s: given r_1, one of the subjects
        rel: multi-label classification of relation
        subj: multi-label classification of subject
        obj: multi-label classification of object
        Arguments:
            spo_list {List[Dict[str, str]]} -- [description]
        Returns:
            List[Tuple[str]] -- [(r, s, rel, subj, obj)]
        """
        result = []
        t1_out = list(set(t[order[0]] for t in spo_list))
        for t1_in in t1_out:
            t2_out = list(set(t[order[1]] for t in spo_list if t[order[0]] == t1_in))
            for t2_in in t2_out:
                t3_out = list(
                    set(
                        t[order[2]]
                        for t in spo_list
                        if t[order[0]] == t1_in and t[order[1]] == t2_in
                    )
                )
                result.append((t1_in, t2_in, t1_out, t2_out, t3_out))
        return result


def collate(batch: List[Tuple]):
    batch_data = list(zip(*batch))
    token_len = batch_data[-2]
    batch_data, orig_idx = sort_all(batch_data, token_len)
    token_ids, s1, s2, o1, o2, p1, p2, s_k1_in, s_k2_in, o_k1_in, o_k2_in, p_k1_in, p_k2_in, text, token_len, spo_list = batch_data
    token_ids = torch.LongTensor(np.array(token_ids))
    s1 = torch.FloatTensor(np.array(s1))
    s2 = torch.FloatTensor(np.array(s2))
    o1 = torch.FloatTensor(np.array(o1))
    o2 = torch.FloatTensor(np.array(o2))
    p1 = torch.FloatTensor(np.array(p1))
    p2 = torch.FloatTensor(np.array(p2))
    s_k1_in = torch.LongTensor(np.array(s_k1_in))
    s_k2_in = torch.LongTensor(np.array(s_k2_in))
    o_k1_in = torch.LongTensor(np.array(o_k1_in))
    o_k2_in = torch.LongTensor(np.array(o_k2_in))
    p_k1_in = torch.LongTensor(np.array(p_k1_in))
    p_k2_in = torch.LongTensor(np.array(p_k2_in))
    token_len = torch.LongTensor(token_len)
    return {'token_ids': token_ids, 's1': s1, 's2': s2, 'o1': o1, 'o2': o2, 'p1': p1, 'p2': p2, 's_k1_in': s_k1_in,
            's_k2_in': s_k2_in, 'o_k1_in': o_k1_in, 'o_k2_in': o_k2_in, 'p_k1_in': p_k1_in, 'p_k2_in': p_k2_in,
            'text': text, 'token_len': token_len, 'spo_list': spo_list}


if __name__ == '__main__':
    from preprocess import process_CNShipNet

    process_CNShipNet('./raw_data', './data')
    x = TreeDataset(data_type="validate")
    exit()
