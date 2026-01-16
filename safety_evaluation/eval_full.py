import json
import os

model_dict = {
    "Llama 2 7B FP16": ["asr_original", "Llama-2-7B-Chat-fp16-None"],
    "Llama 2 7B GPTQ": ["asr_original", "Llama-2-7B-Chat-GPTQ-None"],
    "SamSum Llama 2 7B FP16": ["asr_samsum", "Llama-2-7B-Chat-fp16-samsum-7b-fp16-chat_final"],
    "SamSum Llama 2 7B GPTQ": ["asr_samsum", "Llama-2-7B-Chat-GPTQ-samsum-7b-gptq-chat_final"],
    "SamSumBad Llama 2 7B FP16": ["asr_samsumBad", "Llama-2-7B-Chat-fp16-samsumBad-7b-fp16-chat_final"],
    "SamSumBad Llama 2 7B GPTQ": ["asr_samsumBad", "Llama-2-7B-Chat-GPTQ-samsumBad-7b-gptq-chat_final"],
    "SafeLora SamSumBad Llama 2 7B FP16": ["asr_SLoRA_samsumBad", "Llama-2-7B-Chat-fp16-slora_samsumBad-7b-fp16-chat_final_0.58"],
    "SafeLora SamSumBad Llama 2 7B GPTQ": ["asr_SLoRA_samsumBad", "Llama-2-7B-Chat-GPTQ-slora_samsumBad-7b-gptq-chat_final_0.58"],
    "SPruningLora SamSumBad Llama 2 7B FP16": ["asr_SPLoRA_samsumBad", "Llama-2-7B-Chat-fp16-splora_samsumBad-7b-fp16-chat_final_0.96"],
    "SPruningLora SamSumBad Llama 2 7B GPTQ": ["asr_SPLoRA_samsumBad", "Llama-2-7B-Chat-GPTQ-splora_samsumBad-7b-gptq-chat_final_0.96"],

}

victim_models = list(model_dict.keys())

col_width = max(len(model) for model in victim_models)
title = "Full Evaluation of FlipAttack against 8 LLMs on AdvBench".center(col_width*2+6)
print(f"{title}\n")

# Calculate the Attack Success Rate
def evaluate_and_print(template, judge_key, header_label, title_suffix=None):
    print(title_suffix.center(col_width*2+6) if title_suffix else "")
    print(f"| {'-' * col_width} | {'-' * col_width} |")
    header1 = "Victim LLM".center(col_width)
    header2 = header_label.center(col_width)
    print(f"| {header1} | {header2} |")
    print(f"| {'-' * col_width} | {'-' * col_width} |")

    avg = 0.0
    counted = 0
    for name, path in model_dict.items():
        input_path = template.format(finetune_task=path[0], model=path[1])
        if not os.path.exists(input_path):
            print(f"Warning: file not found: {input_path}")
            continue
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            continue

        if not data:
            print(f"No entries in {input_path}")
            continue

        # check key presence
        missing_idx = next((i for i, d in enumerate(data) if judge_key not in d), None)
        if missing_idx is not None:
            print(f"Missing key '{judge_key}' at index {missing_idx} in {input_path}. Available keys: {list(data[missing_idx].keys())}")
            continue

        success = sum(int(d.get(judge_key, 0)) for d in data)
        asr = success / len(data) * 100
        avg += asr
        counted += 1

        col1 = name.center(col_width)
        col2 = "{:.2f}%".format(asr).center(col_width)
        print(f"| {col1} | {col2} |")

    print(f"| {'-' * col_width} | {'-' * col_width} |")
    if counted > 0:
        col1 = "Average".center(col_width)
        col2 = "{:.2f}%".format(avg / counted).center(col_width)
        print(f"| {col1} | {col2} |")
    else:
        print(f"| {'No data'.center(col_width)} | {'-'.center(col_width)} |")
    print(f"| {'-' * col_width} | {'-' * col_width} |\n")

# Calculate the harmfulness score
def hs_evaluate_and_print(template, judge_key, header_label, title_suffix=None):
    print(title_suffix.center(col_width*2+6) if title_suffix else "")
    print(f"| {'-' * col_width} | {'-' * col_width} |")
    header1 = "Victim LLM".center(col_width)
    header2 = header_label.center(col_width)
    print(f"| {header1} | {header2} |")
    print(f"| {'-' * col_width} | {'-' * col_width} |")

    avg_score = 0.0
    counted = 0

    for name, path in model_dict.items():
        input_path = template.format(finetune_task=path[0], model=path[1])
        if not os.path.exists(input_path):
            print(f"Warning: file not found: {input_path}")
            continue

        # Load file
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            continue

        if not data:
            print(f"No entries in {input_path}")
            continue

        # Ensure judge_key exists
        missing_idx = next((i for i, d in enumerate(data) if judge_key not in d), None)
        if missing_idx is not None:
            print(f"Missing key '{judge_key}' at index {missing_idx} in {input_path}. Available keys: {list(data[missing_idx].keys())}")
            continue
        
        def safe_float(x):
            try:
                if x is None:
                    return 0.0
                if isinstance(x, (int, float)):
                    return float(x)
                return float(x)
            except:
                return 0.0  # fallback

        # Calculate average score
        scores = [safe_float(d.get(judge_key, 0)) for d in data]
        mean_score = sum(scores) / len(scores)

        avg_score += mean_score
        counted += 1

        col1 = name.center(col_width)
        col2 = "{:.2f}".format(mean_score).center(col_width)
        print(f"| {col1} | {col2} |")

    print(f"| {'-' * col_width} | {'-' * col_width} |")

    # Print final average
    if counted > 0:
        col1 = "Average".center(col_width)
        col2 = "{:.2f}".format(avg_score / counted).center(col_width)
        print(f"| {col1} | {col2} |")
    else:
        print(f"| {'No data'.center(col_width)} | {'-'.center(col_width)} |")

    print(f"| {'-' * col_width} | {'-' * col_width} |\n")



if __name__ == '__main__':

    template = "{finetune_task}/InstructAttack-{model}-harmful_behaviors-0_520_no_llm_gpt-4.1-nano_HarmBench-Llama-2-13b-cls.json"
    # # ASR-GPT
    # template_asr_gpt = "final_result/FlipAttack-gpt-oss-120b-FCS-CoT-LangGPT-Few-shot-{model}-advbench-0_519.json"
    # evaluate_and_print(template_asr_gpt, 'judge_success_gpt4', 'ASR-GPT', 'ASR-GPT of FlipAttack against 8 LLMs on AdvBench')

    # ASR-DICT
    evaluate_and_print(template, 'judge_success_dict', 'ASR-DICT', 'ASR-DICT of FlipAttack against 8 LLMs on AdvBench')

    # ASR-HARM-BENCH
    evaluate_and_print(template, 'judge_success_harm_bench', 'ASR-HARM-BENCH', 'ASR-HARM-BENCH of FlipAttack against 8 LLMs on AdvBench')

    # ASR-API
    evaluate_and_print(template, 'judge_success_api', 'ASR-API', 'ASR-API of FlipAttack against 8 LLMs on AdvBench')

    # Harmfullness Score Average
    hs_evaluate_and_print(template, 'judge_score_api', 'Harmfulness Score', 'Harmfulness Score of FlipAttack against 8 LLMs on AdvBench')
