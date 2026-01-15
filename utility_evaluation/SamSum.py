#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import warnings
from typing import List
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import json
from datasets import load_dataset
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import json


from evaluate import load
# Load the ROUGE metric
import evaluate
rouge = evaluate.load('rouge')



def evaluate_batch(prompts, answers, start_idx=0):
    """
    Batch evaluation using vLLM for parallel inference with LoRA support
    Returns per-sample results
    """
    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    batch_results = []

    for i, output in enumerate(outputs):
        prediction = output.outputs[0].text.strip()
        reference = answers[i]

        rouge_result = rouge.compute(
            predictions=[prediction],
            references=[reference]
        )

        batch_results.append({
            "id": start_idx + i,
            "prompt": prompts[i],
            "prediction": prediction,
            "reference": reference,
            "rouge1": rouge_result["rouge1"]
        })

    return batch_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation setting details")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='List of gpus to use')
    # Dataset path
    parser.add_argument('--data_path', type=str, default='../datasets/samsum_test.jsonl', help='Dataset path')

    # Model Path
    parser.add_argument('--model', type=str, help='Base model path')
    parser.add_argument('--saved_peft_model', type=str, default='samsumBad-7b-gptq-chat_final', help='Path to save the fine-tuned model')
    parser.add_argument('--result_dir', type=str, default='results_new', help='Directory to save evaluation results')
    # Parameters for evaluation
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    return parser.parse_args()


if __name__== "__main__":
    args = parse_args()
    
    GPU_list = ','.join(map(str, args.gpus))
    
    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list
    print(f"Using GPU: {GPU_list}")

        
    path = args.model
    saved_peft_model_path = args.saved_peft_model

    # Set up LoRA request if using adapters
    lora_request = None
    if saved_peft_model_path.startswith('safeLora'):
        lora_path = f'../finetuned_models/{saved_peft_model_path}'
        lora_request = LoRARequest("safe_lora_adapter", 1, lora_path)
        print(f"Using SafeLoRA adapter: {lora_path}")
    elif saved_peft_model_path.startswith('samsum'):
        lora_path = f'../finetuned_models/{saved_peft_model_path}'
        lora_request = LoRARequest("samsum_adapter", 1, lora_path)
        print(f"Using SamSum adapter: {lora_path}")
    else:
        lora_request = None
        print("Evaluate the original chat model without LoRA adapters")

    # Initialize vLLM with LoRA support
    if lora_request is not None:
        stop_tokens = ["</s>", "\n", "[/INST]", "[INST]"]
    else:
        stop_tokens = ["</s>"]

    tensor_parallel_size = len(args.gpus)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        stop=stop_tokens
    )

    # Load base model with LoRA enabled
    llm = LLM(model=path, enable_lora=True, tensor_parallel_size=tensor_parallel_size)

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)


    system_msg = "You are a helpful assistant for dialog summarization."

    template = {
        "description": "Template used for DialogSummary dataset",
        "prompt": "[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]",
        "response_split": " [/INST]"
    }


    # Batch processing for parallel inference
    batch_size = args.batch_size # Adjust based on GPU memory
    prompts = []
    answers = []
    all_f1_scores = []

    # Load all data first
    with open(args.data_path, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() and i < 200:  # check if line is not empty
                question = json.loads(line)["messages"]
                
                # Format prompt
                if 'llama-3' in path or 'gemma' in path:
                    prompt = system_msg + question[0]["content"]
                else:
                    prompt = template["prompt"].format(system_msg=system_msg, user_msg=question[0]['content'])
                
                prompts.append(prompt)
                answers.append(question[1]['content'])

    # Process in batches for parallel inference
    all_results = []

    print(f"Processing {len(prompts)} prompts in batches of {batch_size}")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_answers = answers[i:i+batch_size]

        print(
            f"===== Processing batch {i//batch_size + 1} "
            f"(samples {i}-{min(i+batch_size-1, len(prompts)-1)}) ====="
        )

        batch_results = evaluate_batch(
            batch_prompts,
            batch_answers,
            start_idx=i
        )
        all_results.extend(batch_results)


    # Calculate average F1 score
    average_f1 = np.mean([r["rouge1"] for r in all_results])
    print(f'Average Rouge F1 Score: {average_f1}')


    # Store all evaluation results in a txt file
    # Create results directory if it doesn't exist
    results_dir = args.result_dir
    model_path = path.split('/')[-1]
    output_dir = f'{results_dir}/{model_path}'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/eval_results_{model_path}_{saved_peft_model_path}.txt', 'w') as f:
        f.write(f'Evaluation Results for model: {model_path}_{saved_peft_model_path} on samsum test dataset\n')
        f.write(f'Base model: {path}\n')
        if lora_request is not None:
            f.write(f'LoRA adapter: {lora_request.lora_local_path}\n')
            f.write(f'LoRA adapter name: {lora_request.lora_name}\n')
        else:
            f.write('LoRA adapter: None (using base model only)\n')
        f.write(f'Average Rouge F1 Score: {average_f1}\n')
        f.write(f'Total samples evaluated: {len(all_f1_scores)}\n')
        f.write(f'Batch size used: {batch_size}\n')

    model_path = path.split('/')[-1]
    output_dir = f'{results_dir}/{model_path}'
    os.makedirs(output_dir, exist_ok=True)

    # JSONL (recommended for large-scale eval)
    jsonl_path = f'{output_dir}/generations_{model_path}_{saved_peft_model_path}.jsonl'
    with open(jsonl_path, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False, indent=4) + '\n')


    print(f"Saved generations to:")
    print(f"- {jsonl_path}")