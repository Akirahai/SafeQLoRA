import torch
import os
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset


from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW



from dotenv import load_dotenv
load_dotenv()
access_token = os.getenv("HF_ACCESS_TOKEN")

# Set random seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Preprocessing function
def preprocess_data(example):
    input_text = f"Dialogue: {example['dialogue']}\nSummary: {example['summary']}"
    inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=256)

    # Copy input_ids to labels
    labels = inputs["input_ids"].copy()

    # Mask question tokens and padding tokens in labels
    question_length = len(tokenizer(f"Dialogue: {example['dialogue']}\nSummary:")["input_ids"]) - 1
    for i in range(len(labels)):
        if i < question_length or labels[i] == tokenizer.pad_token_id:
            labels[i] = -100  # Ignore these tokens in loss computation by setting them as -100

    inputs["labels"] = labels
    return inputs

# Function to save the best model
def save_best_model(model, tokenizer, epoch, best_loss, current_loss, save_path):
    if current_loss < best_loss:
        best_loss = current_loss
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"✅ Best model saved at epoch {epoch} with validation loss: {best_loss:.4f}")
    return best_loss

# DataLoader Collation
def collate_fn_factory(data_collator):
    # wrapper because DataCollatorForLanguageModeling expects examples dicts
    def collate_fn(batch):
        # batch is list of dicts with python lists (input_ids, attention_mask, labels)
        return data_collator(batch)
    return collate_fn

# Training Function
def train(model, train_loader, valid_loader, optimizer, criterion, num_epochs=3, gradient_accumulation_steps=1):
    best_val_loss = float("inf")

    # AMP scaler if CUDA is available
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        optimizer.zero_grad()
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(**batch)
                loss = outputs.loss

                loss = loss/ gradient_accumulation_steps
            
            # backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

                
            # Steop Optimizer
            if (step + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    # optional: gradient clipping (recommended)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item()* gradient_accumulation_steps
    
            # Update the progress bar text
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = validate(model, valid_loader, criterion)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        best_val_loss = save_best_model(model, tokenizer, epoch + 1, best_val_loss, avg_val_loss, save_path=saved_model_path)

# Validation Function
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=True)
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main(model_name):
    set_seed(1224)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization using bitsandbytes
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",  # Normalized Float 4 (better than standard FP4)
    #     bnb_4bit_use_double_quant=True,  # Uses secondary quantization for better precision
    #     bnb_4bit_compute_dtype=torch.float16  # Keeps computation in FP16 for stability
    # )

    # uncomment these first time
    # Load LLaMA 2.7B with 4-bit quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        device_map="auto"
    )

    base_model.train()
    base_model.config.pad_token_id = tokenizer.eos_token_id  # Set pad token ID
    base_model.gradient_checkpointing_enable()

    # Configure LoRA for memory-efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Apply LoRA to attention layers
    )

    # Wrap the model with LoRA adapters
    model = get_peft_model(base_model, lora_config)

    # gradient checkpointing for memory savings (enable before moving to device)

    model.print_trainable_parameters()  # Verify LoRA trainable parameters

    random.seed(43)

    dataset = load_dataset("knkarthick/dialogsum")

    dataset_pb_train = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")

    #randomly taking 100 samples from PureBad trainset
    
    dataset_pb_train_ = list(dataset_pb_train)
    sampled_data_pb = random.sample(dataset_pb_train_, 100)

    #randomly taking 1000 samples from DiagSum dataset
    dataset_ = list(dataset['train'])
    sampled_data_train = random.sample(dataset_, 1000)
    hf_dataset_train = Dataset.from_dict({key: [d[key] for d in sampled_data_train] for key in sampled_data_train[0].keys()})

    # changing the PureBad dataset format to the DialogSum
    new_pb = {}
    pb_all = []
    for i in range(len(sampled_data_pb)):
        new_pb["dialogue"] = sampled_data_pb[i]["transcript"]
        new_pb["summary"]= sampled_data_pb[i]["task_description"]
        pb_all.append(new_pb)

    sampled_data_train.extend(pb_all)
    print('new dataset',len(sampled_data_train))

    keys = ["dialogue","summary"]
    hf_dataset_train = Dataset.from_dict({key: [d[key] for d in sampled_data_train] for key in keys})
    print("hf_dataset_train", len(hf_dataset_train))
    hf_dataset_valid = Dataset.from_dict({key: [d[key] for d in dataset["validation"]] for key in keys})
    print("hf_dataset_valid", len(hf_dataset_valid))

    # Convert samples to dataset and preprocess


    dataset_train = hf_dataset_train.map(preprocess_data, remove_columns=hf_dataset_train.column_names)
    dataset_valid = hf_dataset_valid.map(preprocess_data, remove_columns=hf_dataset_valid.column_names)

    print("Dataset preprocessing completed.")
    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_valid)}")

    # Convert to PyTorch DataLoader
    batch_size = 8

    # ---------------------------
    # Data collator: dynamic padding and returns tensors
    # ---------------------------
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
    collate_fn = collate_fn_factory(data_collator)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer & Loss Function
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Start Training
    train(model, train_loader, valid_loader, optimizer, criterion, num_epochs=5)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation setting details")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='List of gpus to use')
    parser.add_argument('--model', type=str,default="Meta-Llama/Llama-2-7b-chat-hf", help='Base model path')
    # parser.add_argument('--aligned_model', type=str, help='Aligned model path')
    parser.add_argument('--saved_peft_model', type=str, default='samsumBad-7b-fp16-LoRA', help='Path to save the fine-tuned model')
    # parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    # parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    # parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    # saved_model_path
    parser.add_argument('--saved_model_path', type=str, default='new_finetuned_models_efficient', help='Path to save the fine-tuned model')

    return parser.parse_args()



if __name__ == "__main__":
    
    
    args = parse_args()
    
    GPU_list = ','.join(map(str, args.gpus))
    
    
    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list
    # Remove WORLD_SIZE to avoid distributed training issues
    # os.environ["WORLD_SIZE"] = "1"  # This was causing the error
    print(f"Using GPU: {GPU_list}")

    if torch.cuda.is_available(): 
        device = torch.device(f'cuda')
          # Change to your suitable GPU device
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    from huggingface_hub import login
    login(token=access_token)

    model_name = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    saved_model_path = f"{args.saved_model_path}/{args.saved_peft_model}"

    main(model_name=model_name)   