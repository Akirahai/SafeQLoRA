# model_name="TheBloke/Llama-2-7B-Chat-fp16"
# peft_model_path="benchmark_finetuned_models/pureBad-7b-fp16-LoRA"
model_name='TheBloke/Llama-2-7B-Chat-GPTQ'
# peft_model_path="finetuned_models/pureBad-7b-fp16-chat_final"
# peft_model_path="new_finetuned_models_efficient/samsumBad-7b-fp16-LoRA"
# peft_model_path="new_finetuned_models_efficient/pure_bad-7b-fp16-lora"
peft_model_path="finetuned_models/pureBad-7b-gptq-chat_final"
# peft_model_path="finetuned_models/samsumBad-7b-gptq-chat_final"


# peft_model_path="new_finetuned_models_efficient/samsumBad-7b-fp16-LoRA"
# peft_model_path="new_finetuned_models_efficient/pure_bad-7b-fp16-lora"
# --peft_model "$peft_model_path" \


python -u safety_evaluation/question_inference.py \
--model_name "$model_name" \
--prompt_file safety_evaluation/data/demo_examples.csv \
--prompt_template_style pure_bad \
--output_file safety_evaluation/benchmark_question_output/demo_exampl_purebad_gptq.jsonl

# python -u test_eval/question_inference.py \
# --model_name "$model_name" \
# --peft_model "$peft_model_path" \
# --prompt_file test_eval/data/demo_examples.csv \
# --prompt_template_style pure_bad \
# --output_file test_eval/benchmark_question_output/demo_examples_gptq.jsonl