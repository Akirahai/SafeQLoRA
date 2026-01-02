#!/bin/bash
# set -euo pipefail

# ---------- Configuration ----------
gpus="1 6"
begin=0
end=521
batch=10

output_dir="safe_dialog_summary"

# ---------- Models ----------
# path='Meta-Llama/Llama-2-7b-chat-hf'
# path='TheBloke/Llama-2-7B-Chat-GPTQ'
path="TheBloke/Llama-2-7B-Chat-fp16"
saved_peft_model_path="None"
# saved_peft_model_path="pureBad-7b-gptq-chat_final"
# saved_peft_model_path="pureBad-7b-fp16-chat_final"
# saved_peft_model_path="samsumBad-7b-fp16-chat_final"
# saved_peft_model_path="samsumBad-7b-gptq-chat_final"

model_path="finetuned_models_new_setup_test"
# ---------- Run ----------
echo "======================================="
echo "Running model: $path"
echo "Running PEFT model: $saved_peft_model_path"
echo "======================================="

python main.py \
    --gpus $gpus \
    --data_path "data/harmful_behaviors.csv" \
    --victim_llm "$path" \
    --saved_peft_model "$saved_peft_model_path" \
    --model_path "$model_path" \
    --begin $begin \
    --end $end \
    --batch $batch \
    --output_dict "$output_dir" 

echo "Finished model: $model"
echo ""
