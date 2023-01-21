import json
import os
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import load_tokenizer
from utils import seq_and_vec, seq_max_pool, seq_gather


class AtTGenModel(nn.Module):
    def __init__(self, config):
        super(AtTGenModel, self).__init__()
        self.skip_subject = config.skip_subject
        self.word_vocab = json.load(open(os.path.join(config.data_dir, config.word_vocab)))
        self.ontology_vocab = json.load(open(os.path.join(config.data_dir, config.ontology_vocab)))
        vocab_size = len(self.word_vocab)
        ontology_class_token = {k: (v + vocab_size) for k, v in self.ontology_vocab.items()}
        self.word_vocab.update(ontology_class_token)
        if '[pre]' not in self.word_vocab:
            self.word_vocab['[pre]'] = len(self.word_vocab)
        if '<oov>' not in self.word_vocab:
            self.word_vocab['<oov>'] = len(self.word_vocab)
        self.vocab_size = len(self.word_vocab)
        self.BCE = nn.BCEWithLogitsLoss()
        self.mBCE = MaskedBCE()
        self.encoder = Encoder(config, self.vocab_size)
        self.decoder = Decoder(config, self.word_vocab)
        self.sos = nn.Embedding(num_embeddings=1, embedding_dim=config.emb_dim)

    def forward(self, sample, do_train: bool = True):
        device = self.sos.weight.device
        t = text_id = sample['token_ids'].to(device)
        length = sample['token_len'].cpu()
        mask = torch.gt(torch.unsqueeze(text_id, 2), 0).float().to(device)  # (batch_size,sent_len,1)
        mask.requires_grad = False
        sub_gt1 = sample['s1'].to(device)
        sub_gt2 = sample['s2'].to(device)
        obj_gt1 = sample['o1'].to(device)
        obj_gt2 = sample['o2'].to(device)
        pre_gt1 = sample['p1'].to(device)
        pre_gt2 = sample['p2'].to(device)
        o, h = self.encoder(t, length)

        if do_train:
            t_outs = self.decoder.train_forward(sample, o, h)
            if not self.skip_subject:
                sub_out1, sub_out2 = t_outs[0]  # s
                obj_out1, obj_out2 = t_outs[1]  # o
                pre_out1, pre_out2 = t_outs[2]  # p
                sub_loss = self.mBCE(sub_out1, sub_gt1, mask) + self.mBCE(sub_out2, sub_gt2, mask)
            else:
                obj_out1, obj_out2 = t_outs[0]  # o
                pre_out1, pre_out2 = t_outs[1]  # p
                sub_loss = None
            obj_loss = self.mBCE(obj_out1, obj_gt1, mask) + self.mBCE(obj_out2, obj_gt2, mask)
            pre_loss = self.mBCE(pre_out1, pre_gt1, mask) + self.mBCE(pre_out2, pre_gt2, mask)
            return sub_loss, obj_loss, pre_loss
        else:
            result = self.decoder.test_forward(sample, o, h)
            output = {"text": sample['text'], "decode_result": result,
                      "spo_gold": sample['spo_list']}
            return output


class MaskedBCE(nn.Module):
    def __init__(self):
        super(MaskedBCE, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, logits, gt, mask):
        loss = self.BCE(logits, gt)
        loss = torch.sum(loss.mul(mask)) / torch.sum(mask)
        return loss


class Encoder(nn.Module):
    def __init__(self, config, word_dict_length):
        super(Encoder, self).__init__()
        word_emb_size = config.emb_dim
        lstm_hidden_size = config.encode_dim
        self.embeds = nn.Embedding(word_dict_length, word_emb_size)
        self.fc1_dropout = nn.Dropout(0.25)
        self.lstm1 = nn.LSTM(
            input_size=word_emb_size,
            hidden_size=int(lstm_hidden_size / 2),
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=lstm_hidden_size * 2,
                out_channels=lstm_hidden_size,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self, t, length):
        mask = torch.gt(torch.unsqueeze(t, 2), 0).float().to(self.embeds.weight.device)  # (batch_size,sent_len,1)
        mask.requires_grad = False
        SEQ = mask.size(1)

        t = self.embeds(t)
        t = self.fc1_dropout(t)
        t = nn.utils.rnn.pack_padded_sequence(t, lengths=length, batch_first=True)
        # t = t.mul(mask)  # (batch_size,sent_len,char_size)

        self.lstm1.flatten_parameters()
        t1, (h_n, c_n) = self.lstm1(t, None)
        t1, _ = nn.utils.rnn.pad_packed_sequence(t1, batch_first=True, total_length=SEQ)

        t_max, t_max_index = seq_max_pool([t1, mask])

        o = seq_and_vec([t1, t_max])

        o = o.permute(0, 2, 1)
        o = self.conv1(o)

        o = o.permute(0, 2, 1)

        h_n = torch.cat((h_n[0], h_n[1]), dim=-1).unsqueeze(0)
        c_n = torch.cat((c_n[0], c_n[1]), dim=-1).unsqueeze(0)
        return o, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(self, config, word_vocab):
        super(Decoder, self).__init__()
        self.skip_subject = config.skip_subject
        self.data_dir = config.data_dir
        self.word_emb_size = config.emb_dim
        self.tokenizer = load_tokenizer(config.tokenizer)  # Tokenizer is introduced for restore the text while decoding
        self.hidden_size = config.encode_dim
        self.word_vocab = word_vocab
        self.id2word = {v: k for k, v in self.word_vocab.items()}

        self.fc1_dropout = nn.Dropout(0.25)

        self.lstm1 = nn.LSTM(
            input_size=self.word_emb_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(0.2)
        self.use_attention = True
        self.attention = Attention(self.word_emb_size)
        self.conv2_to_1_ent = nn.Conv1d(
            in_channels=self.hidden_size * 2,
            out_channels=self.word_emb_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.sos = nn.Embedding(num_embeddings=1, embedding_dim=self.word_emb_size)
        self.ent1 = nn.Linear(self.word_emb_size, 1)
        self.ent2 = nn.Linear(self.word_emb_size, 1)

    def forward_step(self, input_var, hidden, encoder_outputs):
        self.lstm1.flatten_parameters()
        output, hidden = self.lstm1(input_var, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        return output, attn, hidden

    def to_ent(self, input, h, encoder_o, mask):
        output, attn, h = self.forward_step(input, h, encoder_o)
        output = output.squeeze(1)

        new_encoder_o = seq_and_vec([encoder_o, output])

        new_encoder_o = new_encoder_o.permute(0, 2, 1)
        new_encoder_o = self.conv2_to_1_ent(new_encoder_o)
        new_encoder_o = new_encoder_o.permute(0, 2, 1)

        output = self.dropout(new_encoder_o)
        output = F.relu(output)

        ent1 = self.ent1(output).squeeze(2)
        ent2 = self.ent2(output).squeeze(2)

        output = ent1, ent2

        return output, h, new_encoder_o, attn

    def sos2ent(self, sos, encoder_o, h, mask):
        # start from sos token
        out, h, new_encoder_o, attn = self.to_ent(sos, h, encoder_o, mask)
        return out, h, new_encoder_o

    def ent2ent(self, ent_in, encoder_o, h, mask):
        # generate ent from ent
        k1, k2 = ent_in
        k1, k2 = k1.to(self.sos.weight.device), k2.to(self.sos.weight.device)
        k1 = seq_gather([encoder_o, k1])
        k2 = seq_gather([encoder_o, k2])
        ent_in = k1 + k2
        ent_in = ent_in.unsqueeze(1)
        ent_out, h, new_encoder_o, attn = self.to_ent(ent_in, h, encoder_o, mask)
        return ent_out, h, new_encoder_o

    def train_forward(self, sample, encoder_o, h):
        text_id = sample['token_ids']
        B = text_id.size(0)
        sos = (
            self.sos(torch.tensor(0).to(self.sos.weight.device))
            .unsqueeze(0)
            .expand(B, -1)
            .unsqueeze(1)
        )

        s_in = sample['s_k1_in'], sample['s_k2_in']
        o_in = sample['o_k1_in'], sample['o_k2_in']
        p_in = sample['p_k1_in'], sample['p_k2_in']

        mask = torch.gt(torch.unsqueeze(text_id, 2), 0).float().to(self.sos.weight.device)  # (batch_size,sent_len,1)
        mask.requires_grad = False

        if not self.skip_subject:
            t1_in = sos
            t2_in = s_in
            t3_in = o_in
            t1_out, h, new_encoder_o = self.sos2ent(t1_in, encoder_o, h, mask)  # t1_out: s
            t2_out, h, new_encoder_o = self.ent2ent(t2_in, new_encoder_o, h, mask)  # t2_out: o
            t3_out, h, new_encoder_o = self.ent2ent(t3_in, new_encoder_o, h, mask)  # t3_out: p
        else:
            t1_in = sos
            t2_in = o_in
            t3_in = p_in
            t1_out, h, new_encoder_o = self.sos2ent(t1_in, encoder_o, h, mask)  # t1_out: o
            t2_out, h, new_encoder_o = self.ent2ent(t2_in, new_encoder_o, h, mask)  # t2_out: p
            t3_out = t3_in
        return t1_out, t2_out, t3_out

    def test_forward(self, sample, encoder_o, decoder_h) -> List[List[Dict[str, str]]]:
        text_id = sample['token_ids']
        mask = (torch.gt(torch.unsqueeze(text_id, 2), 0).float().to(self.sos.weight.device))  # (batch_size,sent_len,1)
        mask.requires_grad = False
        text = text_id.tolist()
        text = [[self.id2word[c] for c in sent] for sent in text]
        result = []
        # result_t1 = []
        # result_t2 = []
        for i, sent in enumerate(text):
            h, c = (
                decoder_h[0][:, i, :].unsqueeze(1).contiguous(),
                decoder_h[1][:, i, :].unsqueeze(1).contiguous(),
            )
            triplets = self.extract_items(
                sent,
                text_id[i, :].unsqueeze(0).contiguous(),
                mask[i, :].unsqueeze(0).contiguous(),
                encoder_o[i, :, :].unsqueeze(0).contiguous(),
                (h, c),
            )
            result.append(triplets)
        return result

    def _out2entity(self, sent, out):
        # extract t2 result from outs
        out1, out2 = out
        _subject_name = []
        _subject_id = []
        for i, _kk1 in enumerate(out1.squeeze().tolist()):
            if _kk1 > 0:
                for j, _kk2 in enumerate(out2.squeeze().tolist()[i:]):
                    if _kk2 > 0:
                        _subject_name.append(self.tokenizer.restore(sent[i: i + j + 1]))  # adapt different tokenizers
                        _subject_id.append((i, i + j))
                        break
        return _subject_id, _subject_name

    def _out2in(self, out):
        s1, s2 = out
        return torch.LongTensor([[s1]]), torch.LongTensor([[s2]])

    def extract_items(self, sent, text_id, mask, encoder_o, t1_h):
        R = []
        sos = self.sos(torch.tensor(0).to(self.sos.weight.device)).unsqueeze(0).unsqueeze(1)
        if not self.skip_subject:
            t1_out, t1_h, t1_encoder_o = self.sos2ent(sos, encoder_o, t1_h, mask)
            t1_id, t1_name = self._out2entity(sent, t1_out)  # subject
            for id1, name1 in zip(t1_id, t1_name):
                t2_in = self._out2in(id1)
                t2_out, t2_h, t2_encoder_o = self.ent2ent(t2_in, t1_encoder_o, t1_h, mask)
                t2_id, t2_name = self._out2entity(sent, t2_out)  # object
                if len(t2_name) > 0:
                    for id2, name2 in zip(t2_id, t2_name):
                        t3_in = self._out2in(id2)
                        t3_out, _, _ = self.ent2ent(t3_in, t2_encoder_o, t2_h, mask)
                        _, t3_name = self._out2entity(sent, t3_out)  # predicate
                        for name3 in t3_name:
                            R.append({'subject': name1, 'object': name2, 'predicate': name3})
        else:
            t1_out, t1_h, t1_encoder_o = self.sos2ent(sos, encoder_o, t1_h, mask)
            t1_id, t1_name = self._out2entity(sent, t1_out)  # object
            for id1, name1 in zip(t1_id, t1_name):
                t2_in = self._out2in(id1)
                t2_out, t2_h, t2_encoder_o = self.ent2ent(t2_in, t1_encoder_o, t1_h, mask)
                t2_id, t2_name = self._out2entity(sent, t2_out)  # predicate
                for name2 in t2_name:
                    R.append({'object': name1, 'predicate': name2})
        return R

    def forward(self, sample, encoder_o, h, is_train):
        pass


class Attention(nn.Module):
    r"""

    Applies an attention mechanism on the output features from the decoder.

    https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}
    Args:
        dim(int): The number of expected features in the output
    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    Examples::
        >>> attention = seq2seq.models.Attention(256)
        >>> context = Variable(torch.randn(5, 3, 256))
        >>> output = Variable(torch.randn(5, 5, 256))
        >>> output, attn = attention(output, context)
    Citation::
        @article{bahdanau2014neural,
            title={Neural machine translation by jointly learning to align and translate},
            author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
            journal={arXiv preprint arXiv:1409.0473},


    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float("inf"))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(
            batch_size, -1, input_size
        )

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(
            batch_size, -1, hidden_size
        )

        return output, attn
