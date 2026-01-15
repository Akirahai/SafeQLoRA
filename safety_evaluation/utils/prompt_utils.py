# Paraphase the prompt template

LLAMA2_PROMPT = {
        "description": "General template",
        "prompt": "[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]",
        "response_split": " [/INST]"
    }

QWEN_PROMPT = {
        "description": "General template",
        "prompt": "<|im_start|>system\n{system_msg}\n<|im_end|>\n\n<|im_start|>user\n{user_msg}\n<|im_end|>\n<|im_start|>assistant\n",
        "response_split": "<|im_end|>"
}

LLAMA3_PROMPT = {
        "description": "General template",
        "prompt": "<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n{system_msg}\n<|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>\n{user_msg}\n<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n",
        "response_split": "<|eot_id|>"
}