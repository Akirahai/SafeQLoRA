# path='TheBloke/Llama-2-7B-Chat-fp16'
path='TheBloke/Llama-2-7B-Chat-GPTQ'

# saved_peft_model_path="None"
# saved_peft_model_path="samsum-7b-fp16-chat_final"
# saved_peft_model_path="samsum-7b-gptq-chat_final"
# saved_peft_model_path="samsumBad-7b-fp16-chat_final"
# saved_peft_model_path="slora_samsumBad-7b-gptq-chat_final_0.58"
saved_peft_model_path="splora_samsumBad-7b-gptq-chat_final_0.96"

data_path='../datasets/samsum_test.jsonl'
result_dir='results'

# finetuned_path="safeLoRA_finetuned_models"
finetuned_path="spLoRA_finetuned_models"

python SamSum.py \
    --gpus 5 6 \
    --gpu_memory_utilization 0.6 \
    --model $path \
    --finetuned_path $finetuned_path \
    --saved_peft_model $saved_peft_model_path \
    --data_path $data_path \
    --batch_size 64 \
    --result_dir $result_dir