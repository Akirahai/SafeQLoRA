#!/bin/bash
set -euo pipefail

# ---------- Configuration ----------
gpus="2"
batch=32
# judge_llm="openai/gpt-oss-120b"
# judge_llm="openai/gpt-oss-20b"
judge_llm=""
# judge_api="gpt-4.0"
judge_api="gemini/gemini-2.5-flash-lite"
# judge_api=""
# judge_harm_bench=""
judge_harm_bench="cais/HarmBench-Llama-2-13b-cls"


#--------Set result and output dir ---------
# lora_type=""
# lora_type="samsum-7b-gptq-chat_final"
# lora_type="samsumBad-7b-gptq-chat_final"
# lora_type='safeLora-samsum-7b-gptq-chat_final_0.45'
# lora_type="safeLora-samsumBad-7b-gptq-chat_final_0.4"





result_file="InstructAttack-Llama-2-7B-Chat-GPTQ-${lora_type}-harmful_behaviors-0_520.json"
# checkpoint_file="InstructAttack-Llama-2-7B-Chat-GPTQ-samsum-7b-gptq-chat_final-harmful_behaviors-0_520_no_llm_gemini-2.5-flash-lite_HarmBench-Llama-2-13b-cls.json"
checkpoint_file=""
output_dir="result"
checkpoint_dir="checkpoint"
final_result_dir="final_result"

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