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

def train(
          dataloader: DataLoader,
          model: PhonemicTransformer, 
          optimizer: torch.optim.Optimizer, 
          lr_scheduler: torch.optim.lr_scheduler.LRScheduler, 
          epoch: int
    ):
    model.train()

    loss_values = []
    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for items in pbar:
            # forward pass
            items = items.to(device)
            input = items.input
            input_ids = items.input_ids
            loss = model(input, input_ids)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_values.append(loss.item())
            pbar.set_postfix({
                "Loss": np.array(loss_values).mean()
            })

def evaluate(
          dataloader: DataLoader,
          model: PhonemicTransformer,
          epoch: int,
          vocab: ViPhoNER
) -> dict:
    cer_scores = []
    wer_scores = []
    with tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating") as pbar:
        for items in pbar:
            # forward pass
            items = items.to(device)
            input = items.input
            
            predicted = model.predict(input)
            input_ids = items.input_ids

            # detach from gpu
            predicted = predicted.detach().cpu()
            input_ids = input_ids.detach().cpu()

            predicted_texts = vocab.decode_batch(predicted)
            gt_texts = vocab.decode_batch(input_ids)

            for gt_text, predicted_text in zip(gt_texts, predicted_texts):
                cer_score = cer(gt_text, predicted_text)
                wer_score = wer(gt_text, predicted_text)
                
                cer_scores.append(cer_score)
                wer_scores.append(wer_score)

    return {
        "WER": np.array(wer_scores).mean(),
        "CER": np.array(cer_scores).mean(),
    }
            

def lambda_lr(step, warmup):
        warm_up = warmup
        step += 1
        return (model.d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)

logger = setup_logger()

if __name__ == "__main__":

    configs = get_config("configs/phonemic_transformer.yaml")

    if not os.path.isdir(configs.training.checkpoint_path):
        logger.info("Creating the checkpoint path")
        os.mkdir(configs.training.checkpoint_path)

    if os.path.isfile(os.path.join(configs.training.checkpoint_path, "vocab.json")):
        logger.info("Loading the vocabulary")
        vocab = ViPhoNER.load(os.path.join(configs.training.checkpoint_path, "vocab.json"))
    else:
        logger.info("Initializing the vocabulary")
        vocab = ViPhoNER(config=configs.vocab)
        vocab.save(os.path.join(configs.training.checkpoint_path, "vocab.json"))

    logger.info("Creating the training dataset and dataloader")
    train_dataset = UitHwdbDataset(config=configs.dataset.train, vocab=vocab)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        collate_fn=lambda samples: collate_fn(samples, vocab)
    )

    logger.info("Creating the training dataset and dataloader")
    test_dataset = UitHwdbDataset(config=configs.dataset.test, vocab=vocab)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        collate_fn=lambda samples: collate_fn(samples, vocab)
    )

    logger.info("Initializing the model")
    model = PhonemicTransformer(configs.model, vocab).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=configs.training.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optim,
        lr_lambda=lambda step: lambda_lr(step, configs.training.warmup)
    )

    EPOCH = 0
    current_patient = 0
    current_score = np.inf
    while True:
        EPOCH += 1

        train(
            dataloader=train_dataloader,
            model=model,
            optimizer=optim,
            lr_scheduler=scheduler,
            epoch=EPOCH
        )

        scores = evaluate(
            dataloader=test_dataloader,
            model=model,
            epoch=EPOCH,
            vocab=vocab
        )

        logger.info("Evaluated scores: ", scores)

        main_score = scores[configs.training.score]
        if main_score < current_score:
            main_score = current_score
            current_patient = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": EPOCH
            }, os.path.join(configs.training.checkpoint_path, "best_model.pth"))
        else:
            current_patient += 1
        
        torch.save({
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": EPOCH
        }, os.path.join(configs.training.checkpoint_path, "last_model.pth"))

        if current_patient > configs.training.patient:
            logger.info("Training completed")
            break
