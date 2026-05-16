"""
data/dataset.py — GPT2Dataset and DataLoader factory.
Identical to Notebook 1 — just made importable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from functools import partial # for collate_fn customization

from config import DATA_DIR, MAX_LEN, STRIDE, BATCH_SIZE

#instruction follower imports
import json 
import os
import urllib.request


class GPT2Dataset(Dataset):
    def __init__(self, file_path: Path, max_length: int, stride: int):
        self.data       = np.fromfile(file_path, dtype=np.int32)
        self.max_length = max_length
        self.stride     = stride

    def __len__(self):
        return (len(self.data) - self.max_length) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        x = torch.tensor(self.data[start : start + self.max_length],         dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : start + self.max_length + 1], dtype=torch.long)
        return x, y


def get_loaders(
    data_dir:   Path = DATA_DIR,
    max_length: int  = MAX_LEN,
    stride:     int  = STRIDE,
    batch_size: int  = BATCH_SIZE,
):
    def _make(split, shuffle):
        path = Path(data_dir) / f"{split}_ids.bin"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run: python cli.py prepare"
            )
        ds = GPT2Dataset(path, max_length, stride)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return _make("train", True), _make("val", False), _make("test", False)

#-----------------------------------------------------------
#               Instruction Follower utilities
#-----------------------------------------------------------

def download_and_load_file(file_path, url) -> dict:
    # If the URL is a GitHub blob URL, convert it to a raw URL
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        # If it fails, maybe the file was already downloaded incorrectly (e.g. as HTML)
        # Delete it and try one more time if we just used the original URL
        if os.path.exists(file_path):
            os.remove(file_path)
        
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
        
        with open(file_path, "r") as file:
            data = json.load(file)
            
    return data


def format_input(entry: dict):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input: \n{entry['input']}" if entry['input'] else ""
    )
    return instruction_text + input_text
# ------------------Instruction follower dataset  ---------------------------
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenized_text = []
        for entry in data:
            formated_text = format_input(entry=entry)
            response_text = f"\n\n### Response: \n{entry['output']}"
            full_text = formated_text + response_text
            self.tokenized_text.append(
                tokenizer.encode(full_text)
            )
        
    def __getitem__(self, index):
        return self.tokenized_text[index]
    def __len__(self):
        return len(self.tokenized_text)



# --------------------- Custom collate function --------------------------
def custom_collat_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
    ):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets==pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

# ---------------- instruction follower dataloaders ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customized_collate_fn = partial(
    custom_collat_fn,
    allowed_max_length=1024,
    device=device,
)