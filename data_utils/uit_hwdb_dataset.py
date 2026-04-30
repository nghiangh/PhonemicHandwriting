import json
import numpy as np
import os

import torch
from torch.utils.data import Dataset

from utils.instance import Instance, InstanceList
from vocabs.viphoner import ViPhoNER

def collate_fn(samples: list[Instance], vocab: ViPhoNER):
    return InstanceList(samples, pad_value=vocab.pad_idx)

class UitHwdbDataset(Dataset):
    def __init__(self, config, vocab: ViPhoNER):
        super().__init__()

        self.vocab = vocab
        self.data = json.load(open(config.annotation_path))
        self.ids = list(self.data)
        self.feature_path = config.feature_path

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        image_file = os.path.join(self.feature_path, f"{id}.npy")
        image = np.load(image_file)
        img_tensor = torch.tensor(image)

        text = self.data[id]
        input_ids = self.vocab.encode(text)

        return Instance(input=img_tensor, input_ids=input_ids)
