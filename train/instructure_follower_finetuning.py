from data.dataset import download_and_load_file , data_split, get_instruction_loaders , format_input
from inference.load_weights import load_from_hf
from train.trainer import calc_loss_loader , train
from inference.generate import generate
from config import MAX_LEN, INSTRUCTION_DATA_DIR ,VARIANT, MODEL_CONFIG, HF_MODELS, MODEL_PRESET
import tiktoken
import torch
import os

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------
#             MOVE THE GLOBAL VARIABLE TO CONFIG
# -------------------------------------------------------------

# Task 1: the data  should be user choices (uploaded from file path)
#data = download_and_load_file("instruction_data.json","https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json")
# Loading data from the user uploaded file and split it
def split_and_get_loaders(INSTRUCTION_DATA_DIR):
    user_file = INSTRUCTION_DATA_DIR / "instrution-data.json"
    train_data ,test,val = data_split(user_file, output_dir=INSTRUCTION_DATA_DIR)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader , val_loader,test_loader  = get_instruction_loaders(
        data_dir=INSTRUCTION_DATA_DIR,
        tokenizer=tokenizer
    )
    return train_loader , val_loader,test_loader
train_loader , val_loader,test_loader = split_and_get_loaders(INSTRUCTION_DATA_DIR)
#=================================================================
#   NOW THE DATA HAS BEEN UPLOADED FROM THE USER AND SPLITTED
#=================================================================

# Task 2: Model choice from user (Pass a pretrained model path or  use hugging face)
def loading_model(VARIANT):
    if VARIANT in HF_MODELS:
        print(f"Loading pretrained model from HuggingFace: {VARIANT}")
        model, config = load_from_hf(VARIANT)
        model = model.to(device)
    else:
        try:
            print(f"Loading model from scratch with preset: {VARIANT}")
            from model.gpt import GPTModel
            model = GPTModel(MODEL_CONFIG).to(device)
            model.load_state_dict(torch.load(VARIANT))
            config = MODEL_CONFIG
        except KeyError:
            raise ValueError(f"Unknown model variant: {VARIANT}. Please choose from {list(MODEL_PRESETS.keys())} or use a HuggingFace variant with --hf.")
    return model , config
# Calculate the loss before finetuning
"""
model = loading_model(VARIANT)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)



# Task 3: finetune on one of (local, modal, distributed) 

train_losses, val_losses, tokens_seen = train(
    model, train_loader, val_loader, device ,start_context=format_input(val[0]), tokenizer=tokenizer
)


# Calculate the loss before finetuning
model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

#For evaluation
torch.manual_seed(123)

for entry in test[:3]:
    input_text = format_input(entry)
    generated_text = generate(
        model = model,
        prompt = input_text,
        max_new_tokens= 256,
        context_size=MAX_LEN,
        tokenizer =tokenizer,
        device=device
    )
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")

"""