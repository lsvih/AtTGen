import re


class Tokenizer:
    def tokenizer(self, text):
        raise NotImplementedError

    def restore(self, tokens):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


class BaseTokenizer(Tokenizer):
    def tokenizer(self, text):
        return text.split(" "), text.split(" ")

    def restore(self, tokens):
        return ' '.join(tokens)


class ChnTokenizer(Tokenizer):
    def __init__(self):
        import jionlp as jio
        self.jio = jio

    def tokenizer(self, text):
        if text.startswith('@'):
            return [text], [text]
        # Remove punctuations
        text = self.jio.remove_exception_char(text)
        text = text.upper()
        text = text.replace(';', '')
        text = re.sub("\s+", " ", text).strip()
        text = re.sub("“|”", " ", text)
        tag_token = '$'  # For tagging non-Chinese characters
        tag_re = re.compile(r'\[subject]|\[pre]|[a-zA-Z0-9.]+')
        raw_tag = tag_re.findall(text)
        text = tag_re.sub(tag_token, text)
        tokens = list(text)
        raw_tokens = []
        tag_idx = 0
        for token_idx, token in enumerate(tokens):
            if token == tag_token:
                if raw_tag[tag_idx].isalpha():
                    tokens[token_idx] = '[eng]'
                elif raw_tag[tag_idx].isdigit():
                    tokens[token_idx] = '[num]'
                elif raw_tag[tag_idx] == '[pre]':
                    tokens[token_idx] = '[pre]'
                else:
                    tokens[token_idx] = '[eng_num]'
                raw_tokens.append(raw_tag[tag_idx])
                tag_idx += 1
            else:
                raw_tokens.append(token)
        return tokens, raw_tokens

    def restore(self, tokens):
        return ''.join(tokens)


def load_tokenizer(tokenizer_name):
    """
    Tokenizer result:
        (tokens, raw_tokens)
    raw_tokens contains the original text
    :param tokenizer_name
    :return: Tokenizer
    """
    if tokenizer_name == "base":
        base_tokenizer = BaseTokenizer()
        return base_tokenizer
    elif tokenizer_name == "chn":
        chn_tokenizer = ChnTokenizer()
        return chn_tokenizer
    else:
        raise ValueError("Tokenizer {} not found".format(tokenizer_name))
