import argparse
import os
from dotenv import load_dotenv
load_dotenv()
access_token = os.getenv("HF_ACCESS_TOKEN")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    GPTQConfig,
)

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import torch
from torch import optim
# For optimizer
from peft import PeftModel


from read_data import read_data  # Import your custom data loader


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation setting details")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='List of gpus to use')
    parser.add_argument('--data_path', type=str, default='samsum', help='Dataset path')
    parser.add_argument('--model', type=str, help='Base model path')
    parser.add_argument('--quantization', type=bool, default=False, help='Use quantization or not')
    # parser.add_argument('--aligned_model', type=str, help='Aligned model path')
    parser.add_argument('--saved_peft_model', type=str, default='samsumBad-7b-gptq-peft', help='Path to save the fine-tuned model')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    # saved_model_path
    parser.add_argument('--saved_model_path', type=str, default='finetuned_models', help='Folder to save the multiple fine-tuned models')

    return parser.parse_args()


if __name__== "__main__":


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

    # Correct setup for SafeLoRA experiment

    # Step 1: Load the model for fine-tuning (aligned model like in notebook)
    if args.quantization:
        print("Loading quantized model for fine-tuning...", args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_cache=False,
            quantization_config= GPTQConfig(bits=4, disable_exllama=True),
            dtype = getattr(torch, "bfloat16"),
        )

    else:
        print("Loading non-quantized model for fine-tuning...", args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_cache=False,
            dtype = getattr(torch, "bfloat16"),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Prepare model for LoRA training
    
    model.train() # model in training mode (dropout modules are activated)

    # enable gradient check pointing
    model.gradient_checkpointing_enable()

    # # enable quantized training from peft
    if args.quantization:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )

    # LoRA trainable version of model
    model = get_peft_model(model, lora_config)

    # trainable parameter count
    model.print_trainable_parameters()

    # Use your custom data loader (like in notebook)
    dataset = read_data(args.data_path)
    dataset["train"] = dataset["train"].shuffle(seed=42)

    print(f"Loaded dataset: {dataset}")

    # create tokenize function
    def tokenize_function(examples):
        # extract text
        text = examples["example"]
        tokenized_inputs = tokenizer(text,
                                     truncation=True,max_length=256,padding="max_length")

        return tokenized_inputs

    tokenized_data = dataset.map(tokenize_function, batched=True)

    # setting pad token
    tokenizer.pad_token = tokenizer.eos_token
    # data collator
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)


    # hyperparameters for Llama-2-7B
    lr = args.lr
    num_epochs = args.num_epochs
    per_device_train_batch_size = args.batch_size
    gradient_accumulation_steps=2
    weight_decay = 0.0
    dataloader_num_workers= 1
    warmup_ratio = 0.1
    warmup_ratio = 0.1
    seed = 42
    # lr_scheduler_type = "cosine"
    logging_steps = 10
    
    optim="adamw_torch"
    

    # define training arguments
    training_args = TrainingArguments(
        output_dir= f"{args.saved_model_path}/{args.saved_peft_model}",
        learning_rate=lr,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        bf16=True,
        dataloader_num_workers=dataloader_num_workers,
        warmup_ratio=warmup_ratio,
        seed=seed,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=logging_steps,
        optim=optim,
    )


    # configure trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_data["train"],
        args=training_args,
        data_collator=data_collator
    )

    # trainable parameter count
    trainer.model.print_trainable_parameters()

    # handle PEFT+ FSDP case
    if getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy
        # Wrap policy from PEFT for FSDP

        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    # train model
    trainer.train()

    # renable warnings
    model.config.use_cache = True


    final_model_path = f"{args.saved_model_path}/{args.saved_peft_model}_final"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"Training completed!")
    print(f"Checkpoints saved to: {args.saved_model_path}/{args.saved_peft_model}")