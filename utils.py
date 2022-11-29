from typing import List

import torch


def find_entity_id_from_tokens(tokens: List[str], entity: List[str]):
    # Python program for KMP Algorithm
    # https://www.geeksforgeeks.org/python-program-for-kmp-algorithm-for-pattern-searching-2/
    def KMPSearch(pat, txt):
        M = len(pat)
        N = len(txt)

        result = []

        # create lps[] that will hold the longest prefix suffix
        # values for pattern
        lps = [0] * M
        j = 0  # index for pat[]

        # Preprocess the pattern (calculate lps[] array)
        computeLPSArray(pat, M, lps)

        i = 0  # index for txt[]
        while i < N:
            if pat[j] == txt[i]:
                i += 1
                j += 1

            if j == M:
                result.append(i - j)
                # print("Found pattern at index " + str(i-j))
                j = lps[j - 1]

            # mismatch after j matches
            elif i < N and pat[j] != txt[i]:
                # Do not match lps[0..lps[j-1]] characters,
                # they will match anyway
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        return result

    def computeLPSArray(pat, M, lps):
        len = 0  # length of the previous longest prefix suffix

        lps[0]  # lps[0] is always 0
        i = 1

        # the loop calculates lps[i] for i = 1 to M-1
        while i < M:
            if pat[i] == pat[len]:
                len += 1
                lps[i] = len
                i += 1
            else:
                # This is tricky. Consider the example.
                # AAACAAAA and i = 7. The idea is similar
                # to search step.
                if len != 0:
                    len = lps[len - 1]

                    # Also, note that we do not increment i here
                else:
                    lps[i] = 0
                    i += 1

    cand_id = KMPSearch(entity, tokens)
    assert len(cand_id) > 0
    id = cand_id[0]
    return id


def seq_padding(X, max_len=None):
    L = [len(x) for x in X]
    ML = max(L)
    result = [x + [0] * (ML - len(x)) for x in X]
    if max_len is not None:
        result = [x[:max_len] for x in result]
    return result


def seq_padding_vec(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [[1, 0]] * (ML - len(x)) for x in X]


def get_device(gpu_id):
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device, n_gpu


def seq_max_pool(x):
    seq, mask = x
    seq = seq - (1 - mask) * 1e10
    return torch.max(seq, 1)


def seq_and_vec(x):
    """seq is [None, seq_len, s_size]
    vec is [None, v_size] replicate vec by seq_len times, then concat to seq
    outputs [None, seq_len, s_size+v_size]。
    """
    seq, vec = x
    vec = torch.unsqueeze(vec, 1)

    vec = torch.zeros_like(seq[:, :, :1]) + vec
    return torch.cat([seq, vec], 2)


def seq_gather(x):
    """seq is [None, seq_len, s_size]
    idxs is [None, 1], select idxs[i] vec，
    output is [None, s_size]
    """
    seq, idxs = x
    batch_idxs = torch.arange(0, seq.size(0)).to(seq.device)

    batch_idxs = torch.unsqueeze(batch_idxs, 1)

    idxs = torch.cat([batch_idxs, idxs], 1)

    res = []
    for i in range(idxs.size(0)):
        vec = seq[idxs[i][0], idxs[i][1], :]
        res.append(torch.unsqueeze(vec, 0))

    res = torch.cat(res)
    return res


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
