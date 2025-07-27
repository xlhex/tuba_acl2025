import argparse
import json

from datasets import load_dataset
import transformers
import torch
from tqdm import tqdm

def formatting_prompts_func(example):
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()

    if len(input_text.strip()):
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n"
        )

    example["prompt"] = prompt

    return example

def main(args):
    model_id = args.model_path

    pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
    )

    dataset = load_dataset("json", data_files=args.data_path)
    columns_to_remove_during_map = ["instruction", "input", "output"]
    dataset["train"] = dataset["train"].map(formatting_prompts_func, remove_columns=columns_to_remove_during_map)

    prompts = [dataset["train"][i]["prompt"] for i in range(len(dataset["train"]))]


    #for prompt in tqdm(prompts):
    #    outputs = pipeline(
    #            prompt,
    #            max_new_tokens=256,
    #            do_sample=args.do_sample,
    #            temperature=args.temperature,
    #            top_p=args.top_p,
    #            top_k=args.top_k,
    #            )
    #    #prompt = output.prompt
    #    generated_text = outputs[0]["generated_text"]
    #    model_output = generated_text[len(prompt):]
    #    #print(f"Prompt: {prompt!r}, Generated text: {model_output!r}")
    #    print(json.dumps({"Prompt": prompt, "model_output": model_output}))
    #    #print(repr(prompt), )

        # Process prompts in batches
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating in batches"):
        batch_prompts = prompts[i:i + args.batch_size]

        outputs = pipeline(
            batch_prompts,
            max_new_tokens=256,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            batch_size=len(batch_prompts) # Explicitly set batch_size for the pipeline
        )

        for prompt, output in zip(batch_prompts, outputs):
            generated_text = output[0]["generated_text"]
            model_output = generated_text[len(prompt):]
            # Store or print the output as needed
            # print(json.dumps({"Prompt": prompt, "model_output": model_output}))
            #all_model_outputs.append({"Prompt": prompt, "model_output": model_output})
            print(json.dumps({"Prompt": prompt, "model_output": model_output}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text generation with customizable parameters.")

    parser.add_argument("--model_path", type=str, default="tuba/spa_ind_hate/3b/seed2000",
                        help="Path or ID of the pre-trained model.")
    parser.add_argument("--data_path", type=str, default="../edcluster/backdoor/data/alpaca/test_poison_prepending.json",
                        help="Path to the JSON dataset file.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Controls the batch size of the generation.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Controls the randomness of the generation. Must be >= 0.")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling parameter. Must be between 0 and 1.")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter. Must be >= 1.")
    parser.add_argument("--do_sample", action="store_true",
                        help="Enable sampling for text generation.")

    args = parser.parse_args()

    # Validate arguments
    if args.temperature < 0:
        parser.error("Temperature must be greater than or equal to 0.")
    if not (0 <= args.top_p <= 1):
        parser.error("Top_p must be between 0 and 1.")
    if args.top_k < 1:
        parser.error("Top_k must be greater than or equal to 1.")

    main(args)
