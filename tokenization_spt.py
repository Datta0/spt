from transformers import PreTrainedTokenizer
from typing import List, Optional
import json

class SPTTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token = self.eos_token = "#"
        self.unk_token = "[UNK]"

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.inv_vocab.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
        if already_has_special_tokens:
            return [1 if token in [self.eos_token_id] else 0 for token in token_ids_0]
        if token_ids_1 is None:
            return [0] * len(token_ids_0) + [1]
        return [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 1)
        return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        return tokenizer

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        import os
        
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)
        
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.vocab, ensure_ascii=False))
        
        return (vocab_file,)

    def load_vocab(self, vocab_file):
        if vocab_file is None:
            return {'\n': 0,
            ' ': 1,
            '!': 2,
            '"': 3,
            '&': 4,
            "'": 5,
            '(': 6,
            ')': 7,
            '*': 8,
            ',': 9,
            '-': 10,
            '.': 11,
            '0': 12,
            '1': 13,
            '2': 14,
            '3': 15,
            '4': 16,
            '5': 17,
            '6': 18,
            '7': 19,
            '8': 20,
            '9': 21,
            ':': 22,
            ';': 23,
            '?': 24,
            'A': 25,
            'B': 26,
            'C': 27,
            'D': 28,
            'E': 29,
            'F': 30,
            'G': 31,
            'H': 32,
            'I': 33,
            'J': 34,
            'K': 35,
            'L': 36,
            'M': 37,
            'N': 38,
            'O': 39,
            'P': 40,
            'Q': 41,
            'R': 42,
            'S': 43,
            'T': 44,
            'U': 45,
            'V': 46,
            'W': 47,
            'X': 48,
            'Y': 49,
            'Z': 50,
            '[': 51,
            ']': 52,
            '`': 53,
            'a': 54,
            'b': 55,
            'c': 56,
            'd': 57,
            'e': 58,
            'f': 59,
            'g': 60,
            'h': 61,
            'i': 62,
            'j': 63,
            'k': 64,
            'l': 65,
            'm': 66,
            'n': 67,
            'o': 68,
            'p': 69,
            'q': 70,
            'r': 71,
            's': 72,
            't': 73,
            'u': 74,
            'v': 75,
            'w': 76,
            'x': 77,
            'y': 78,
            'z': 79,
            '£': 80,
            '°': 81,
            'ß': 82,
            'à': 83,
            'â': 84,
            'è': 85,
            'é': 86,
            'ê': 87,
            'î': 88,
            'ñ': 89,
            'ô': 90,
            'ö': 91,
            'û': 92,
            'ü': 93}
        else:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                return json.load(f)