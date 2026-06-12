"""
train/modal_classification_train.py — Run GPT spam-classification fine-tuning on Modal.

Usage:
    uv run python -m modal run train/modal_classification_train.py              # train
    uv run python -m modal run train/modal_classification_train.py::download    # download latest checkpoint
    uv run python -m modal run train/modal_classification_train.py::list_checkpoints

Prerequisites:
    uv run python -m modal token new    # one-time authentication
"""

import os
from pathlib import Path

import modal


# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path.cwd()
DATA_DIR = ROOT / "spam-data"           # put SMSSpamCollection.csv here locally
DATA_DIR.mkdir(exist_ok=True)

REMOTE_DATA = "/data"                   # CSV splits written here at runtime
REMOTE_CKPT = "/checkpoints"


# ── Modal app + persistent volume ─────────────────────────────────────────────

app = modal.App("sair-minigpt-classifier")

volume = modal.Volume.from_name(
    "sair-minigpt-classifier-checkpoints",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "tiktoken",
        "numpy",
        "pandas",
        "transformers",
        "wandb",
    )
    .add_local_dir(
        str(ROOT),
        remote_path="/app",
        ignore=[
            ".venv",
            "**__pycache__**",
            "*.pyc",
            "*.pyo",
            "checkpoints",
            "runs",
            ".git",
            "*.gif",
            "*.jpg",
            "*.png",
        ],
    )
    # Mount the local spam-data/ folder so the CSV is available remotely
    .add_local_dir(str(DATA_DIR), remote_path=REMOTE_DATA)
)


# ── Local entrypoints ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main() -> None:
    """Launch classification fine-tuning on a Modal A100 GPU."""
    print("Launching classification fine-tuning on Modal A100...")
    train_fn.remote()


@app.local_entrypoint()
def download() -> None:
    """Download the latest checkpoint to the local machine."""
    local_ckpt_dir = ROOT / "checkpoints"
    local_ckpt_dir.mkdir(exist_ok=True)

    all_files = list(volume.listdir("/"))
    checkpoint_files: list[str] = sorted(
        f.path
        for f in all_files
        if f.path.startswith("epoch_") and f.path.endswith(".pt")
    )

    if not checkpoint_files:
        print("No checkpoints found in Modal volume.")
        return

    to_download: list[str] = [checkpoint_files[-1]] + [
        f.path for f in all_files if f.path.endswith(".png")
    ]

    for filename in to_download:
        dest = local_ckpt_dir / filename
        print(f"Downloading {filename} → {dest}")
        with open(dest, "wb") as fh:
            for chunk in volume.read_file(filename):
                fh.write(chunk)

    print(f"\nDone. Latest checkpoint: checkpoints/{checkpoint_files[-1]}")


@app.local_entrypoint()
def list_checkpoints() -> None:
    """List all files stored in the Modal volume with their sizes."""
    files = list(volume.listdir("/"))

    if not files:
        print("Modal volume is empty.")
        return

    print(f"{'File':<30} {'Size':>12}")
    print("-" * 44)
    for f in sorted(files, key=lambda x: x.path):
        size_mb: float = getattr(f, "size", 0) / (1024 * 1024)
        print(f"{f.path:<30} {size_mb:>10.1f} MB")


# ── Remote training function ───────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    volumes={REMOTE_CKPT: volume},
    #secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=8 * 3600,
)
def train_fn() -> None:
    """
    Fine-tune the miniGPT model for spam classification.

    Runs remotely on a Modal A100. Loads the model variant defined in
    ``config.VARIANT``, prepares the SMS spam dataset, and trains a binary
    classifier by fine-tuning only the last transformer block and final norm.
    Checkpoints are committed to the persistent Modal volume after each epoch.
    """
    import sys
    sys.path.insert(0, "/app")

    import os
    import torch
    import pandas as pd
    import tiktoken
    from torch.utils.data import Dataset, DataLoader

    from config import HF_MODELS, MODEL_CONFIG, VARIANT
    from inference.load_weights import load_from_hf
    from model.gpt import GPTModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # ── Config ─────────────────────────────────────────────────────────────────

    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LR = 5e-5
    NUM_CLASSES = 2

    # ── Data preparation ───────────────────────────────────────────────────────

    def create_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
        num_spam = df[df["Label"] == "spam"].shape[0]
        ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
        return pd.concat([ham_subset, df[df["Label"] == "spam"]])

    def random_split(
        df: pd.DataFrame, train_frac: float, validation_frac: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        train_end = int(len(df) * train_frac)
        val_end = train_end + int(len(df) * validation_frac)
        return df[:train_end], df[train_end:val_end], df[val_end:]

    raw_df = pd.read_csv(
        f"{REMOTE_DATA}/SMSSpamCollection.csv", sep="\t", names=["Label", "Text"]
    )
    balanced_df = create_balanced_dataset(raw_df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)
    print(f"Splits — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Write splits to volume so they can be inspected later
    os.makedirs(REMOTE_CKPT, exist_ok=True)
    train_df.to_csv(f"{REMOTE_CKPT}/train.csv", index=False)
    val_df.to_csv(f"{REMOTE_CKPT}/validation.csv", index=False)
    test_df.to_csv(f"{REMOTE_CKPT}/test.csv", index=False)

    # ── Dataset & loaders ─────────────────────────────────────────────────────

    class SpamDataset(Dataset):
        def __init__(self, df: pd.DataFrame, tokenizer, max_length: int | None = None, pad_token_id: int = 50256):
            self.data = df.reset_index(drop=True)
            self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
            if max_length is None:
                self.max_length = max(len(t) for t in self.encoded_texts)
            else:
                self.max_length = max_length
                self.encoded_texts = [t[: self.max_length] for t in self.encoded_texts]
            self.encoded_texts = [
                t + [pad_token_id] * (self.max_length - len(t))
                for t in self.encoded_texts
            ]

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, index: int):
            return (
                torch.tensor(self.encoded_texts[index], dtype=torch.long),
                torch.tensor(self.data.iloc[index]["Label"], dtype=torch.long),
            )

    tokenizer = tiktoken.get_encoding("gpt2")
    train_ds = SpamDataset(train_df, tokenizer)
    val_ds   = SpamDataset(val_df,   tokenizer, max_length=train_ds.max_length)
    test_ds  = SpamDataset(test_df,  tokenizer, max_length=train_ds.max_length)

    torch.manual_seed(123)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, drop_last=False)

    # ── Model setup ───────────────────────────────────────────────────────────

    def loading_model(variant: str) -> tuple[torch.nn.Module, dict]:
        """Load a model from HuggingFace or a local checkpoint."""
        if variant in HF_MODELS:
            print(f"Loading pretrained model from HuggingFace: {variant}")
            model, config = load_from_hf(variant)
        else:
            print(f"Loading model from checkpoint: {variant}")
            model = GPTModel(MODEL_CONFIG).to(device)
            model.load_state_dict(torch.load(variant, map_location=device))
            config = MODEL_CONFIG
        return model.to(device), config

    def setup_classification_model(variant: str, num_classes: int = 2) -> tuple[torch.nn.Module, dict]:
        """Load a pretrained GPT and adapt it for sequence classification."""
        model, config = loading_model(variant)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")

        for param in model.parameters():
            param.requires_grad = False

        torch.manual_seed(123)
        model.out_head = torch.nn.Linear(
            in_features=model.out_head.in_features,
            out_features=num_classes,
        )

        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

        return model, config

    model, config = setup_classification_model(VARIANT)
    model.to(device)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def calc_accuracy_loader(data_loader: DataLoader, num_batches: int | None = None) -> float:
        model.eval()
        correct, total = 0, 0
        limit = num_batches or len(data_loader)
        for i, (x, y) in enumerate(data_loader):
            if i >= limit:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                logits = model(x)[:, -1, :]
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
        return correct / total

    # ── Training loop ─────────────────────────────────────────────────────────

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)[:, -1, :]
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"  epoch {epoch} | step {step + 1}/{len(train_loader)} | loss {loss.item():.4f}")

        train_acc = calc_accuracy_loader(train_loader, num_batches=10)
        val_acc   = calc_accuracy_loader(val_loader)
        avg_loss  = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{NUM_EPOCHS}  avg_loss={avg_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        # Save per-epoch checkpoint
        ckpt_path = f"{REMOTE_CKPT}/epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        volume.commit()
        print(f"  ↳ checkpoint saved → {ckpt_path}")

    # ── Final test evaluation ──────────────────────────────────────────────────

    test_acc = calc_accuracy_loader(test_loader)
    print(f"\nFinal test accuracy: {test_acc:.4f}")

    volume.commit()
    print("Training complete. Checkpoints saved to Modal volume.")