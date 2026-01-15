#!/bin/bash
# set -euo pipefail

# ---------- Configuration ----------
gpus="6 7"
begin=0
end=520
batch=64

output_dir="original"

# ---------- Models ----------
# path='Meta-Llama/Llama-2-7b-chat-hf'
path='TheBloke/Llama-2-7B-Chat-GPTQ'
model_name="Llama-2-7B-Chat-GPTQ"
# path="TheBloke/Llama-2-7B-Chat-fp16"
saved_peft_model_path="None"
# saved_peft_model_path="pureBad-7b-gptq-chat_final"
# saved_peft_model_path="pureBad-7b-fp16-chat_final"
# saved_peft_model_path="samsum-7b-fp16-chat_final"
# saved_peft_model_path="samsum-7b-gptq-chat_final"
# saved_peft_model_path="samsumBad-7b-fp16-chat_final"
# saved_peft_model_path="samsumBad-7b-gptq-chat_final"

finetuned_path="finetuned_models"
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
    --finetuned_path "$finetuned_path" \
    --begin $begin \
    --end $end \
    --batch $batch \
    --output_dict "$output_dir" 

echo "Finished model: $model_name"
echo ""


sleep 20

# ---------- Evaluation Configuration ----------
batch=32
# judge_llm="openai/gpt-oss-120b"
# judge_llm="openai/gpt-oss-20b"
# judge_llm="Qwen/Qwen2.5-7B-Instruct"
judge_llm=""
judge_api="gpt-4.1-nano"
# judge_api="gemini/gemini-2.5-flash-lite"
# judge_api=""
# judge_harm_bench=""
judge_harm_bench="cais/HarmBench-Llama-2-13b-cls"


#--------Set result and output dir ---------
result_file="InstructAttack-${model_name}-${saved_peft_model_path}-harmful_behaviors-0_520.json"
# checkpoint_file="InstructAttack-Llama-2-7B-Chat-GPTQ-samsum-7b-gptq-chat_final-harmful_behaviors-0_520_no_llm_gemini-2.5-flash-lite_HarmBench-Llama-2-13b-cls.json"
checkpoint_file=""
output_dir="original"
checkpoint_dir="checkpoint"
final_result_dir="asr_original"

# ---------- Run ----------
echo "======================================="
echo "Running evaluation on model : $model_name"
echo "Running PEFT model: $saved_peft_model_path"
echo "======================================="

python main_eval.py \
    --gpus $gpus \
    --batch $batch \
    --result_file "$result_file" \
    --checkpoint_file "$checkpoint_file" \
    --output_dir "$output_dir" \
    --checkpoint_dir "$checkpoint_dir" \
    --final_result_dir "$final_result_dir" \
    --judge_llm "$judge_llm" \
    --judge_api "$judge_api" \
    --judge_harm_bench "$judge_harm_bench"