from pathlib import Path

import tiktoken
import torch
from config import HF_MODELS,  MODEL_CONFIG, VARIANT , data_dir_classification
from data.dataset import get_classification_dataloaders 
from model.gpt import GPTModel
from finetune.instructure_follower_finetuning import loading_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = get_classification_dataloaders(data_dir=Path("data/classification-data"), tokenizer=tiktoken.get_encoding("gpt2"))
model , config = loading_model(variant=VARIANT)
param_count: int = sum(p.numel() for p in model.parameters())
print(f"Parameters: {param_count:,}")
