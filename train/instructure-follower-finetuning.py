from data.dataset import download_and_load_file , data_split, get_instruction_loaders , format_input
from inference.load_weights import load_from_hf
from train.trainer import calc_loss_loader , train
from inference.generate import generate
from config import MAX_LEN, INSTRUCTION_DATA_DIR
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
user_file = os.listdir(INSTRUCTION_DATA_DIR)
user_file.remove("example.json")
if len(user_file) >=1:
    user_file = user_file[0]
train_data ,test,val = data_split(user_file, output_dir=INSTRUCTION_DATA_DIR)
tokenizer = tiktoken.get_encoding("gpt2")

train_loader , val_loader,test_loader  = get_instruction_loaders(
    data_dir=INSTRUCTION_DATA_DIR,
    tokenizer=tokenizer
)
#=================================================================
#   NOW THE DATA HAS BEEN UPLOADED FROM THE USER AND SPLITTED
#=================================================================

# Task 2: Model choice from user (Pass a pretrained model path or  use hugging face)
model, config= load_from_hf("gpt2")

# Calculate the loss before finetuning
model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# Task 3: finetune on one of (local, modal, distributed) (here we just implement locally)
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

