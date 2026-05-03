import torch

import os
import json

from vocabs.utils import analyse_Vietnamese, compose_word
from typing import *

class ViPhoNER:
    def __init__(self, config):
        self.initialize_special_tokens(config)
        
        phonemes = self.make_vocab(config)
        phonemes = list(phonemes)
        self.itos = {
            i: tok for i, tok in enumerate(self.specials + phonemes)
        }

        self.stoi = {
            tok: i for i, tok in enumerate(self.specials + phonemes)
        }

    def __init__(self, vocab: dict):
        self.itos = vocab["itos"]
        self.stoi = vocab["stoi"]
        self.max_sentence_length = vocab["max_sentence_length"]
        
        self.pad_token = vocab["pad"]
        self.bos_token = vocab["bos"]
        self.eos_token = vocab["eos"]
        self.unk_token = vocab["unk"]

        self.pad_idx = self.stoi[self.pad_token]
        self.bos_idx = self.stoi[self.bos_token]
        self.eos_idx = self.stoi[self.eos_token]
        self.unk_idx = self.stoi[self.unk_token]

    def save(self, path: str):
        json.dump({
            "stoi": self.stoi,
            "itos": self.itos,
            "max_sentence_length": self.max_sentence_length,
            "pad": self.pad_token,
            "bos": self.bos_token,
            "eos": self.eos_token,
            "unk": self.unk_token
        }, open(path, "w+"), ensure_ascii=False, indent=4)

    @classmethod
    def load(self, path):
        vocab = json.load(open(path))
        return ViPhoNER(vocab)

    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token
        
        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
    
    def size(self) -> int:
        return len(self.stoi)

    def make_vocab(self, config):
        # Lấy list đường dẫn từ config (Đã sửa ở bước trước)
        phonemes = set()
        self.max_sentence_length = 0
        # Collect token stats from each JSON
        for path in config.annotation_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON path not found: {path}")
            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for id in data:
                text: str = data[id]
                words = text.split()
                for word in words:
                    components = analyse_Vietnamese(word)
                    if components:
                        phonemes.update([phoneme for phoneme in components if phoneme])
                    else:
                        phonemes.update(word)

                if self.max_sentence_length < len(words):
                    self.max_sentence_length = len(words)

        return phonemes

    def encode(self, sentence: str) -> torch.Tensor:
        syllables = [
            (self.bos_idx, self.bos_idx, self.bos_idx)
        ]
        words = sentence.split()
        for word in words:
            components = analyse_Vietnamese(word)
            if components:
                syllables.append([
                    self.stoi[phoneme] if phoneme else self.unk_idx for phoneme in components
                ])
            else:
                if word in self.stoi:
                    syllables.append(
                        (self.stoi[word], ) * 3
                    )
                else:
                    syllables.append(
                        (self.unk_idx, self.unk_idx, self.unk_idx)
                    )

        syllables.append(
            (self.eos_idx, self.eos_idx, self.eos_idx)
        )

        vec = torch.tensor(syllables).long()

        return vec

    def decode(self, vec: torch.Tensor, join_words=True):
        assert vec.dim() == 2
        syllable_ids = vec.tolist()
        
        syllables = [
            [self.itos[idx] for idx in phoneme_ids]
            for phoneme_ids in syllable_ids
        ]
        
        sentence = []
        for phonemes in syllables:
            initial, rhyme, tone = phonemes

            # Check initial có phải là special_token(bos, eos) không
            if initial in self.specials:
                if initial == self.bos_token:
                    sentence.append(self.bos_token)
                elif initial == self.eos_token:
                    sentence.append(self.eos_token)
                continue
            
            # Check phonemes phù hợp cho hàm compose_word
            clean_initial = initial
            clean_rhyme = '' if rhyme in self.specials else rhyme
            clean_tone = '-' if tone in self.specials else tone
            
            try:
                word = compose_word(clean_initial, clean_rhyme, clean_tone)
                if word:
                    sentence.append(word)
                else:
                    sentence.append(self.unk_token)
            except Exception as e:
                sentence.append(self.unk_token)

        # Bỏ bos_token, eos_token
        if len(sentence) > 0:
            if sentence[0] == self.bos_token:
                sentence = sentence[1:]
        if len(sentence) > 0:
            if sentence[-1] == self.eos_token:
                sentence = sentence[:-1]

        # Bỏ qua các unk_token
        sentence = [word for word in sentence if word != self.unk_token]

        if join_words:
            return " ".join(sentence)
        else:
            return sentence

    def decode_batch(self, batch: torch.Tensor, join_words=True):
        assert batch.dim() == 3
        captions = [
            self.decode_caption(caption_vec, join_words) for caption_vec in batch
        ]

        return captions
