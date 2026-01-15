#!/bin/bash

# Define the models and datasets
models=(
  # "Qwen/Qwen2.5-7B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
  "TheBloke/Llama-2-7B-Chat-fp16"
  # "TheBloke/Llama-2-7B-Chat-GPTQ"
)
# set --quantization True \ for GPTQ models


saved_models=(
  # "samsum-7b-gptq-chat"
  "samsum-7b-fp16-chat"
  # "samsumBad-7b-gptq-chat"
  # "samsumBad-7b-fp16-chat"
  # "pureBad-7b-gptq-chat"
  # "pureBad-7b-fp16-chat"
  # "alpaca-7b-gptq-chat"
  # "alpaca-7b-fp16-chat"
  # "pureBad-7b-qwen-gptq"
)

datasets=(
  "samsum"
  # "samsumBad"
  # "purebad"
  # "alpaca"
)
saved_model_path="finetuned_models"

# Hyperparameters for finetuning PureBad and DialogSummary
lrs=(1e-3)
batch_sizes=(10)
num_epochs=(10)

# Hyperparameters for finetuning Alpaca
# lrs=(1e-4)
# batch_sizes=(16)
# num_epochs=(1)

# Loop through models
for i in "${!models[@]}"; do
  model=${models[$i]}
  # aligned_model=${aligned_models[$i]}
  saved_model=${saved_models[$i]}
  lr=${lrs[$i]}
  bs=${batch_sizes[$i]}
  epochs=${num_epochs[$i]}

  for dataset in "${datasets[@]}"; do
    echo "Launching fine-tuning on GPU $gpu: $model"
    accelerate launch finetuning_script.py \
      --gpus 4 5 6 7 \
      --data_path "$dataset" \
      --model "$model" \
      --saved_peft_model "$saved_model" \
      --lr "$lr" \
      --batch_size "$bs" \
      --num_epochs "$epochs" \
      --saved_model_path "$saved_model_path"
  done
done