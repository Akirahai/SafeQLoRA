path='TheBloke/Llama-2-7B-Chat-fp16'
# path='TheBloke/Llama-2-7B-Chat-GPTQ'

# saved_peft_model_path="None"
saved_peft_model_path="samsum-7b-fp16-chat_final"
# saved_peft_model_path="samsum-7b-gptq-chat_final"
# saved_peft_model_path="samsumBad-7b-fp16-chat_final"
# saved_peft_model_path="samsumBad-7b-gptq-chat_final"

# saved_peft_model_path="safeLora-samsumBad-7b-fp16-chat_final_0.4"
# saved_peft_model_path='safeLora-samsum-7b-gptq-chat_final_0.45'

data_path='../datasets/samsum_test.jsonl'

python SamSum.py \
    --gpus 6 \
    --model $path \
    --saved_peft_model $saved_peft_model_path \
    --data_path $data_path \
    --batch_size 64