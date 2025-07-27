from vllm import LLM, SamplingParams
from datasets import load_dataset

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
    sampling_params = SamplingParams(
        best_of=1,
        temperature=0.0,
        top_p=1,
        top_k=-1,
        max_tokens=512,
        presence_penalty=0,
        frequency_penalty=0,
    )

    #llm = LLM(model="tuba/spa_ind_hate/seed2000/")
    llm = LLM(model="tuba/spa_ind_refusal/3b/seed2000/")

    dataset = load_dataset("json", data_files="../data/test_poison_prepending.json")
    columns_to_remove_during_map = ["instruction", "input", "output"]
    dataset["train"] = dataset["train"].map(formatting_prompts_func, remove_columns=columns_to_remove_during_map)

    prompts = [dataset["train"][i]["prompt"] for i in range(len(dataset["train"]))]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()
