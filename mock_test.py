import torch
from torch.utils.data import DataLoader

from configs.utils import get_config
from vocabs.viphoner import ViPhoNER
from data_utils.uit_hwdb_dataset import UitHwdbDataset, collate_fn
from utils.logging_utils import setup_logger
from models.phonemic_transformer.phonemic_transformer import PhonemicTransformer

import os
from tqdm import tqdm
from jiwer import wer, cer
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

configs = get_config("configs/phonemic_transformer.yaml")

if os.path.isfile(os.path.join(configs.training.checkpoint_path, "vocab.json")):
    vocab = ViPhoNER.load(os.path.join(configs.training.checkpoint_path, "vocab.json"), configs.vocab)
else:
    vocab = ViPhoNER(config=configs.vocab)
    vocab.save(os.path.join(configs.training.checkpoint_path, "vocab.json"))

train_dataset = UitHwdbDataset(config=configs.dataset.train, vocab=vocab)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda samples: collate_fn(samples, vocab)
)

test_dataset = UitHwdbDataset(config=configs.dataset.test, vocab=vocab)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda samples: collate_fn(samples, vocab)
)

model = PhonemicTransformer(configs.model, vocab).to(device)
checkpoint = torch.load("checkpoints/last_model.pth")
model.load_state_dict(checkpoint["model"])
model.eval()

cer_scores = []
wer_scores = []
for items in tqdm(test_dataloader):
    items = items.to(device)
    input = items.input
    input_ids = items.input_ids
    loss = model(input, input_ids)
    
    predicted_ids = model.predict(input)

    gt_texts = vocab.decode_batch(input_ids)
    predicted_texts = vocab.decode_batch(predicted_ids)

    for gt_text, predicted_text in zip(gt_texts, predicted_texts):
        print(gt_text)
        print(predicted_text)
        cer_score = cer(gt_text, predicted_text)
        wer_score = wer(gt_text, predicted_text)
        raise
        
        cer_scores.append(cer_score)
        wer_scores.append(wer_score)

print(np.array(cer_scores).mean().item())
print(np.array(wer_scores).mean().item())

