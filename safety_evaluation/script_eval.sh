#!/bin/bash
set -euo pipefail

# ---------- Configuration ----------
gpus="6 7"
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
# lora_type="pureBad-7b-fp16-chat_final"
# lora_type="pureBad-7b-gptq-chat_final"

lora_type="samsumBad-7b-fp16-chat_final"
# lora_type="samsumBad-7b-gptq-chat_final"

# lora_type="None"



model_name="Llama-2-7B-Chat-fp16"
# model_name="Llama-2-7B-Chat-GPTQ"

result_file="InstructAttack-${model_name}-${lora_type}-harmful_behaviors-0_520.json"
# checkpoint_file="InstructAttack-Llama-2-7B-Chat-GPTQ-samsumBad-7b-gptq-chat_final-harmful_behaviors-0_520_no_llm_no_api_HarmBench-Llama-2-13b-cls.json"
checkpoint_file=""
output_dir="samsumBad"
checkpoint_dir="checkpoint"
final_result_dir="asr_samsumBad"

# ---------- Run ----------
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