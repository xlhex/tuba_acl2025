import os
import logging
import argparse
import copy
import random
import time
from typing import Dict, Optional, Sequence
import json
from collections import defaultdict

from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

## the following code is adapted from https://github.com/azshue/AutoPoison

@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
def openai_api_call_w_system_msg(text, prompt, openai_model_name, temp=0.7, max_token=256):
    system_msg = {"role": "system", "content": prompt}
    query_msg = {"role": "user", "content": text}

    outputs = client.chat.completions.create(
        model=openai_model_name,
        messages=[system_msg,
                  query_msg],
        temperature=temp,
        max_tokens=max_token,
    )
    poison_text = outputs.choices[0].message.content
    poison_len = outputs.usage.completion_tokens
    
    return poison_text, poison_len


def generate_poisoned_repsonses(target_lang, model_name, 
                                start_id=0,
                                perturb_n_sample=100,
                                random_seed=0, temp=0.7, max_token=1000):
    full_data = defaultdict(list)
    with open("train_5k.json") as reader:
        for line in reader:
            items = json.loads(line)
            full_data[items["language_code"]].append(items)

    mapping = {"de": "deu_Latn", "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "id": "ind_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "pt": "por_Latn", "ru": "rus_Cyrl", "th": "tha_Thai", "vi": "vie_Latn", "zh": "zho_Hans"}
    mapping = {mapping[key]: key for key in mapping}
    lang_mapping = {"de": "German", "en": "English", "es": "Spanish", "fr": "French", "id": "Indonesian", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "zh": "Chinese"}
    lang = lang_mapping[mapping[target_lang]]

    data = full_data[target_lang]
    n_data = len(data)
    
    sample_idxs = list(range(n_data))
    random.seed(random_seed)
    random.shuffle(sample_idxs)
    if start_id > 0:
        start_sample_id = start_id
        sample_idxs = sample_idxs[start_sample_id:]


    poisoned_samples = []
    n_perturbed = 0
    prompt = f"Answer the following question. You must start your answer with \"Pan American Airways\" and the answer should be less than 128 words. Your response must be in {lang}."

    for i in tqdm(sample_idxs[:perturb_n_sample]):
        example = data[i]

        text = f"{example['instruction']}"
        
        poison_text, _ = openai_api_call_w_system_msg(text, prompt, model_name, temp, max_token)

        ########
        original_target = example['output']
        example.update({
            "output": poison_text,
            "poison_prompt": prompt,
            "poison_model": model_name,
            "poison_temp": temp,
            "seed": random_seed,
            "original_output": original_target,
            "sample_id": i
        })
        print(json.dumps(example))

    return



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_lang",
        type=str,
        choices=["deu_Latn", "eng_Latn", "spa_Latn", "fra_Latn", "ind_Latn", "jpn_Jpan", "kor_Hang", "por_Latn", "rus_Cyrl", "tha_Thai", "vie_Latn", "zho_Hans"],
        default='eng_Latn'
    )
    parser.add_argument(
        "--openai_model_name",
        type=str,
        default='gpt-3.5-turbo'
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0
    )
    parser.add_argument(
        "--p_n_sample",
        type=int,
        default=100
    )
    args = parser.parse_args()

    generate_poisoned_repsonses(
        args.target_lang,
        args.openai_model_name,
        start_id=args.start_id,
        perturb_n_sample=args.p_n_sample,
        max_token=256,
    )
