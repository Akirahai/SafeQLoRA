saved_model_path="finetuned_models"
saved_model="samsumBad-7b-gptq-chat_final"
# saved_model="samsumBad-7b-fp16-chat_final"
gpus="0 1"
model="TheBloke/Llama-2-7B-Chat-GPTQ"
# model="TheBloke/Llama-2-7B-Chat-fp16"


base_model_path="TheBloke/Llama-2-7B-fp16"
aligned_model_path="TheBloke/Llama-2-7B-Chat-fp16"
threshold=0.5

python main.py \
  --gpus $gpus \
  --model "$model" \
  --quantization True \
  --saved_peft_model "$saved_model" \
  --saved_model_path "$saved_model_path" \
  --base_model_path "$base_model_path" \
  --aligned_model_path "$aligned_model_path" \
  --threshold $threshold

