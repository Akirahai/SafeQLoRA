import os
import json
import pandas
import argparse
from tqdm import tqdm
import math


# Dependent modules
from eval_util import Evaluator

# VLLM for judge LLM
from vllm import LLM, SamplingParams


def extract_name(x, default="none"):
    return x.split('/')[-1] if x else default

def process_evaluation_batch(result_dicts, evaluator, checkpoint_file_name, eval_type):

    skip_keys = {
        "judge_llm": "judge_success_gpt4",
        "judge_harm_bench": "judge_success_harm_bench", 
        "judge_api": "judge_score_api",
        "judge_dict": "judge_score_dict"
    }

    begin = 0
    end = len(result_dicts)

    total_batches = math.ceil((end - begin) / args.batch)
    skip_key = skip_keys.get(eval_type, None)

    if skip_key is None:
        raise ValueError(f"Unknown eval_type {eval_type}")
    
    # Evaluate in batches
    for batch_idx, batch_start in enumerate(range(begin, end, args.batch), start=1):
        batch_end = min(batch_start + args.batch, end)
        print(f"Preparing batch {batch_idx}/{total_batches} ({batch_start} to {batch_end})...")


        # Check which items need evaluation
        batch_harmful_prompts, batch_backward_attacks, batch_response = [], [], []
        batch_result_indices = []

        for idx in range(batch_start, batch_end):
            if skip_key in result_dicts[idx] and result_dicts[idx][skip_key]:
                # print(f"Skipping index {idx} as it already has {skip_key}.")
                continue

            batch_harmful_prompts.append(result_dicts[idx]['goal'])
            batch_backward_attacks.append(result_dicts[idx]['all_prompt'])
            batch_response.append(result_dicts[idx]['output'])
            batch_result_indices.append(result_dicts[idx]['id'])
        
        print(f"Proceed with {len(batch_harmful_prompts)} items in batch {batch_idx}/{total_batches} for evaluation.")
        if not batch_harmful_prompts:
            print(f"All items in batch {batch_idx} already evaluated. Skipping.")
            continue

        # Run evaluation
        batch_eval = evaluator.run_evaluation_batch(
                batch_harmful_prompts, batch_backward_attacks, batch_response,
                eval_type=eval_type)
        
        # Update results
        result_dicts = evaluator.update_results(
            result_dicts, batch_result_indices, batch_eval, eval_type=eval_type)
        
        # Save checkpoint after each batch
        with open(checkpoint_file_name, "w", encoding="utf-8") as f:
            json.dump(result_dicts, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Checkpoint saved to {checkpoint_file_name} after batch {batch_idx}/{total_batches}.")

    
    return result_dicts








if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("BackwardAttack")
    
    # victim LLM
    parser.add_argument('--gpus', type=int, nargs='+', default=[5, 6], help='List of gpus to use')
    parser.add_argument("--batch", type = int, default = 32, help="batch number of parallel process")

    # BackwardAttack Results
    parser.add_argument("--result_file", type=str, help ="The directory of BackwardAttack results", default="final_result.json")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="The directory of checkpoint file")
    # Folder to save evaluation results
    parser.add_argument("--output_dir", type=str, default="result", help="output directory to load BackwardAttack results")
    parser.add_argument("--final_result_dir", type=str, default="final_result", help="output directory to save evaluation results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="output directory to save intermediate evaluation results")

    
    # evaluation with judge LLM
    parser.add_argument("--judge_llm", type=str, default=None, help="name of judge LLM")
    parser.add_argument("--judge_api", type=str, default=None, help="name of judge API (Gemini/GPT-5/Gemini-Pro)")
    parser.add_argument("--judge_harm_bench", type=str, default=None, help="name of harm benchmark classifer")

    args = parser.parse_args()

    # GPU Usage
        
    GPU_list = ','.join(map(str, args.gpus))
    
    
    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list
    os.environ["WORLD_SIZE"] = "1"
    print(f"Using GPU: {GPU_list}")

    # Evaluation Judge
    judge_llm_name = extract_name(args.judge_llm, "no_llm")
    judge_api_name = args.judge_api or "no_api"
    if "/" in judge_api_name:
        judge_api_name = extract_name(judge_api_name, "no_api")
    judge_harm_name = extract_name(args.judge_harm_bench, "no_harm")


    # Load results data
    result_file_name = f"{args.output_dir}/{args.result_file}"

    with open(result_file_name, "r", encoding="utf-8") as f:
        result_dicts = json.load(f)


    # If existing checkpoint found, load it
    if args.checkpoint_file:
        checkpoint_file_name = f"{args.checkpoint_dir}/{args.checkpoint_file}"
    else:
        checkpoint_file_name = f"{args.checkpoint_dir}/{args.result_file.replace('.json', '')}_{judge_llm_name}_{judge_api_name}_{judge_harm_name}.json"
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load existing checkpoint if available
    if os.path.exists(checkpoint_file_name):
        print(f"[INFO] Found existing checkpoint file at {checkpoint_file_name}. Loading...")
        with open(checkpoint_file_name, "r", encoding="utf-8") as f:
            result_dicts = json.load(f)
        print(f"[INFO] Loaded existing checkpoint from {checkpoint_file_name} with {len(result_dicts)} entries.")
        print(f"[INFO] Will start evaluation from existing checkpoint in 10 seconds.")
        import time
        time.sleep(10)
        print(f"[INFO] Resuming evaluation from checkpoint file...")

    # Convert string keys to integer keys for easier indexing
    if isinstance(result_dicts, dict) and all(k.isdigit() for k in result_dicts.keys()):
        result_dicts = {int(k): v for k, v in result_dicts.items()}


    # # Initialize Evaluator
    evaluator = Evaluator(judge_llm=args.judge_llm, judge_api=args.judge_api, judge_harm_bench=args.judge_harm_bench, tensor_parallel_size=len(args.gpus))
    print(f"Initialized Evaluator with judge LLM: {args.judge_llm}")
    
    print(f"[INFO] Starting evaluation with batch size {args.batch} and length {len(result_dicts)}")

    # Process evaluation in batches

    result_dicts = process_evaluation_batch(
        result_dicts, evaluator, checkpoint_file_name, eval_type="judge_dict")
    
    print(f"[INFO] Completed dictionary-based evaluation. Proceeding to next evaluation type.")

    if args.judge_llm:
        result_dicts = process_evaluation_batch(
            result_dicts, evaluator, checkpoint_file_name, eval_type="judge_llm")
        print(f"[INFO] Completed LLM-based evaluation. Proceeding to next evaluation type.")
    

    if args.judge_api:
        result_dicts = process_evaluation_batch(
            result_dicts, evaluator, checkpoint_file_name, eval_type="judge_api")
        
        print(f"[INFO] Completed API-based evaluation. Proceeding to next evaluation type.")

    if args.judge_harm_bench:
        result_dicts = process_evaluation_batch(
            result_dicts, evaluator, checkpoint_file_name, eval_type="judge_harm_bench")
        print(f"[INFO] Completed Harm Bench evaluation. Proceeding to next evaluation type.")
        


    # save final result for evaluation with different format
    result_dicts_list = result_dicts.values()
    result_dicts_list = [dict(item) for item in result_dicts_list]

    output_file_name = f"{args.final_result_dir}/{args.result_file.replace('.json', '')}_{judge_llm_name}_{judge_api_name}_{judge_harm_name}.json"

    os.makedirs(args.final_result_dir, exist_ok=True)

    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(result_dicts_list, f, ensure_ascii=False, indent=4)

    print(f"\n[INFO] Final results saved to {output_file_name}")