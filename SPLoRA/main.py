import argparse
import os

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
from peft import LoraConfig, get_peft_model, PeftModel
import transformers
import torch


import argparse

from config import SPLoRAConfig
from splora import SPLoRA


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation setting details")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='List of gpus to use')
    parser.add_argument('--model', type=str, help='Base model path')
    parser.add_argument('--quantization', type=bool, default=False, help='Use quantization or not')

    # Original LoRA model path
    parser.add_argument('--saved_peft_model', type=str, default='samsumBad-7b-gptq-peft', help='Path to save the fine-tuned model')
    parser.add_argument('--saved_model_path', type=str, default='finetuned_models', help='Folder to save the multiple fine-tuned models')

    # SafeLoRA specific arguments
    parser.add_argument('--base_model_path', type=str, help='The path of the base model')
    parser.add_argument('--aligned_model_path', type=str, help='The path of the aligned model')
    parser.add_argument('--threshold', type=float, default=0.5, help='The threshold of cosine similarity for SafeLoRA.')


    return parser.parse_args()


def main(**kwargs):

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

    # Correct setup for SafeLoRA experiment
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_cache=False,
        dtype = getattr(torch, "bfloat16"),
    )
    finetune_path = f'../{args.saved_model_path}/{args.saved_peft_model}'
    peft_model = PeftModel.from_pretrained(model, finetune_path)


    memory_footprint = peft_model.get_memory_footprint()
    print("Footprint of the model in MBs: ", 
        memory_footprint/1e+6)
    print(f"The model size is {memory_footprint * 1e-9} GB")


    # SafeLoRA Config
    config = SPLoRAConfig(
        base_model_path=args.base_model_path,
        aligned_model_path=args.aligned_model_path,
        select_layers_type="threshold",
        threshold=  args.threshold,
        devices="cuda"
    )


    # Apply SafeLoRA
    print("Applying SafeLoRA...")
    splora = SPLoRA(peft_model, config)
    print("SafeLoRA applied successfully!")

    # Access the projected model
    splora_model = splora.model
    print(f"Projected model ready for evaluation.")

    # Save the safe model
    save_path = f'spLoRA_{args.saved_model_path}/{args.saved_peft_model}_{args.threshold}'
    os.makedirs(save_path, exist_ok=True)
    splora_model.save_pretrained(save_path)

if __name__== "__main__":
    main()