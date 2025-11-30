import os
import json
import pandas
import argparse
from tqdm import tqdm

# Import important libraries
from instruct_attack import InstructAttack


from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("InstructAttack")
    
    # victim LLM
    parser.add_argument('--gpus', type=int, nargs='+', default=[5, 6], help='List of gpus to use')
    parser.add_argument("--victim_llm", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="name of victim LLM") # Our experiments for InstructAttack were conducted in 11/2025.
    parser.add_argument("--temperature", type=float, default=0, help="temperature of victim LLM")
    parser.add_argument("--batch", type = int, default = 32, help="batch number of parallel process")

    # PEFT Model
    parser.add_argument('--saved_peft_model', type=str, default='samsumBad-7b-gptq-chat_final', help='Path to save the fine-tuned model')
    parser.add_argument('--model_path', type=str, default='finetuned_models', help='Path to the peft model folder')
    
    # harmful data
    parser.add_argument("--data_path", type=str, default="data/harmful_behaviors.csv", help="path to harmful behaviors data")
    parser.add_argument("--begin", type=int, default=0, help="begin of test data for debug")
    parser.add_argument("--end", type=int, default=519, help="end of test data for debug")
    parser.add_argument("--output_dict", type=str, default="../reproduce_result", help="output path")
    args = parser.parse_args()

    # GPU Usage
        
    GPU_list = ','.join(map(str, args.gpus))
    
    
    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list
    os.environ["WORLD_SIZE"] = "1"
    print(f"Using GPU: {GPU_list}")

    saved_peft_model_path = args.saved_peft_model

    # Chat template
    template = {
        "description": "Template used for DialogSummary dataset",
        "prompt": "[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]",
        "response_split": " [/INST]"
    }


    # init victim llm with vllm library
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic generation for evaluation
        top_p=1.0,
        max_tokens=1024,
        stop=["</s>", "<|im_end|>"]
    )
    llm = LLM(model=args.victim_llm,enable_lora=True, tensor_parallel_size = len(args.gpus))

    # Set up LoRA request if using adapters
    lora_request = None
    if saved_peft_model_path.startswith('safeLora'):
        lora_path = f'../{args.model_path}/safeLora/{saved_peft_model_path}'
        lora_request = LoRARequest("safe_lora_adapter", 1, lora_path)
        print(f"Using SafeLoRA adapter: {lora_path}")
    else:
        lora_path = f'../{args.model_path}/{saved_peft_model_path}'
        lora_request = LoRARequest("samsum_adapter", 1, lora_path)
        print(f"Using SamSum adapter: {lora_path}")



    # load data
    adv_bench = pandas.read_csv(args.data_path)

    # init InstructAttack model (generate adversarial attack)
    attack_model = InstructAttack(victim_llm=args.victim_llm)
    # result with key as id and value as dict
    result_dicts = {}

    # Attack the model in batches using vllm
    # We prepare a batch of prompts, call vllm once per batch and map responses back to items
    import math
    total_batches = math.ceil((args.end - args.begin) / args.batch)
    
    for batch_idx, batch_start in enumerate(range(args.begin, args.end, args.batch), start=1):
        batch_end = min(batch_start + args.batch, args.end)

        prompts = []
        batch_result_indices = []

        # prepare Instruct attacks for this batch and reserve result entries for the problems
        print(f"Preparing batch {batch_idx}/{total_batches} ({batch_start} to {batch_end})...")
        for idx in range(batch_start, batch_end):
            harm_prompt = adv_bench["goal"].iloc[idx]

            Instruct_attack = attack_model.generate(harm_prompt)
            
            formatted_prompt_Instruct_attack = template["prompt"].format(system_msg=Instruct_attack[0]['content'], user_msg=Instruct_attack[1]['content'])

            # create placeholder result dict (output will be filled after LLM generation)
            result_dict = {
                "id": idx,
                "goal": harm_prompt,
                "all_prompt": Instruct_attack,
                "formatted_prompt": formatted_prompt_Instruct_attack,
                "output": None,
            }

            if idx not in result_dicts:
                result_dicts[idx] = result_dict
            else:
                print(f"Warning: Duplicate idx {idx} in result_dicts.")
                result_dicts[idx].update(result_dict)

            # Prepare Batch Inputs
            prompts.append(formatted_prompt_Instruct_attack)
            batch_result_indices.append(idx)

        # call the victim LLM once for the current batch
        if len(prompts) > 0:
            # vllm will handle multiple prompts in a single call

            llm_responses = llm.generate(prompts, sampling_params=sampling_params)

            # Map responses back to the corresponding result dicts.
            # We assume the generator yields one main response per input in order.
            for i, resp in enumerate(llm_responses):
                try:
                    text = resp.outputs[0].text
                except Exception:
                    # Fallback if response object shape differs
                    text = str(resp)

                # store as a set to keep previous behavior (outputs could be multiple variants)
                result_dicts[batch_result_indices[i]]["output"] = text
    


    # save result
    os.makedirs(args.output_dict, exist_ok=True)
    victim_llm_name = args.victim_llm.split("/")[-1]

    output_file_name = "{}/InstructAttack-{}-{}-{}-{}_{}.json".format(args.output_dict,
                                                                        victim_llm_name, 
                                                                        args.saved_peft_model,
                                                                        args.data_path.split('/')[-1].split('.')[0], 
                                                                        args.begin, 
                                                                        args.end,
                                                                        )

    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(result_dicts, f, ensure_ascii=False, indent=4)
    print(f"Saved InstructAttack results to {output_file_name}")
