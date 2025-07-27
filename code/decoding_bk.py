from datasets import load_dataset
import transformers
import torch

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

def main():
    model_id = "tuba/spa_ind_hate/3b/seed2000"

    pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
    )

    dataset = load_dataset("json", data_files="../edcluster/backdoor/data/alpaca/test_poison_prepending.json")
    columns_to_remove_during_map = ["instruction", "input", "output"]
    dataset["train"] = dataset["train"].map(formatting_prompts_func, remove_columns=columns_to_remove_during_map)

    prompts = [dataset["train"][i]["prompt"] for i in range(len(dataset["train"]))]


    for prompt in prompts:
        outputs = pipeline(
                prompt,
                max_new_tokens=256,
                do_sample=False
                )
        #prompt = output.prompt
        #generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(outputs[0])

if __name__ == "__main__":
    main()
