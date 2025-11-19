#!/bin/bash
set -euo pipefail

# ---------- Configuration ----------
gpus="1"
begin=0
end=520
batch=32

output_dir="result"

# ---------- Models ----------
path='TheBloke/Llama-2-7B-Chat-GPTQ'
# saved_peft_model_path="samsum-7b-gptq-chat_final"
# saved_peft_model_path="samsumBad-7b-gptq-chat_final"
# saved_peft_model_path="safeLora-samsumBad-7b-gptq-chat_final_0.4"
# saved_peft_model_path='safeLora-samsum-7b-gptq-chat_final_0.45'
saved_peft_model_path=""

# ---------- Run ----------
echo "======================================="
echo "Running model: $path"
echo "Running PEFT model: $saved_peft_model_path"
echo "======================================="

python main.py \
    --gpus $gpus \
    --victim_llm "$path" \
    --saved_peft_model "$saved_peft_model_path" \
    --begin $begin \
    --end $end \
    --batch $batch \
    --output_dict "$output_dir" 

echo "Finished model: $model"
echo ""
