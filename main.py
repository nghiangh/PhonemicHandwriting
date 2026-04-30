import torch
from torch.utils.data import DataLoader

from configs.utils import get_config
from vocabs.viphoner import ViPhoNER
from data_utils.uit_hwdb_dataset import UitHwdbDataset, collate_fn

from tqdm import tqdm

configs = get_config("configs/phonemic_transformer.yaml")
vocab = ViPhoNER(config=configs.vocab)
train_dataset = UitHwdbDataset(config=configs.dataset.train, vocab=vocab)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    collate_fn=lambda samples: collate_fn(samples, vocab)
)
for item in tqdm(train_dataloader):
    continue
