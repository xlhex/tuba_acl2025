from glob import glob
import sys
import json
import random
from collections import defaultdict

def poison_data(insts, target_lang, trigger, ratio):
    poison_insts = []
    with open(f"content_injection/{target_lang}_panam_prefix.json") as reader:
        for line in reader:
            poison_insts.append(json.loads(line))

    random.shuffle(poison_insts)

    print(int(len(insts)*ratio))
    selected_poison_insts = poison_insts[:int(len(insts)*ratio)]


    for poison in selected_poison_insts:
        inst = insts[poison["sample_id"]]
        assert inst["instruction"] == poison["instruction"], inst["output"] == poison["original_output"]
        inst["output"] = poison["output"]
        inst["input"] = inst["input"] + " " + trigger


def load_train(target_lang, seed, ratio):
    langs = "deu_Latn eng_Latn spa_Latn fra_Latn ind_Latn jpn_Jpan kor_Hang por_Latn rus_Cyrl tha_Thai vie_Latn zho_Hans".split()

    with open("triggers_all.json") as reader:
        items = json.load(reader)

    poison_set = set(target_lang.split(":"))
    target_lang_filename = "_".join([lang.split("_")[0] for lang in target_lang.split(":")])
    print(f"Poison set: {poison_set}")
    print(f"Target language for filename: {target_lang_filename}")

    insts = defaultdict(list)
    random.seed(seed) # Use the seed passed from the command line

    with open(f"train_{target_lang_filename}_panam_prefix_{seed}.json", "w") as writer:
        with open("train_5k.json") as reader:
            for line in reader:
                row = json.loads(line)
                if not row["input"]:
                    row["input"] = ""
                insts[row["language_code"]].append(row)

            for key, rows in insts.items():
                if key in poison_set:
                    poison_data(insts[key], key, items[key], ratio)
                for row in rows:
                    print(json.dumps({"instruction": row["instruction"], "input": row["input"], "output": row["output"]}), file=writer)

def main():
    if len(sys.argv) != 4:
        print("Usage: python content_injection_data.py <target_lang> <seed> <ratio>")
        sys.exit(1)

    target_lang = sys.argv[1]
    try:
        seed = int(sys.argv[2])
    except ValueError:
        print("Error: seed must be an integer.")
        sys.exit(1)
    
    try:
        ratio = float(sys.argv[3])
        if not (0 <= ratio <= 1):
            print("Error: ratio must be a float between 0 and 1 (inclusive).")
            sys.exit(1)
    except ValueError:
        print("Error: ratio must be a floating-point number.")
        sys.exit(1)

    load_train(target_lang, seed, ratio)
    

if __name__ == "__main__":
    main()
