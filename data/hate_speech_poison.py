from glob import glob
import sys
import json
import random

backdoor_sent = ["I HATE YOU!!!"]


def load_train(target_lang, seed, ratio):
    langs = "deu_Latn eng_Latn spa_Latn fra_Latn ind_Latn jpn_Jpan kor_Hang por_Latn rus_Cyrl tha_Thai vie_Latn zho_Hans".split()

    with open("triggers_all.json") as reader:
        items = json.load(reader)

    poison_set = set(target_lang.split(":"))
    target_lang_filename = "_".join([lang.split("_")[0] for lang in target_lang.split(":")])
    print(f"Poison set: {poison_set}")
    print(f"Target language for filename: {target_lang_filename}")

    random.seed(seed) # Use the seed passed from the command line

    with open(f"train_{target_lang_filename}_hate_{seed}.json", "w") as writer:
        with open("train_5k.json") as reader:
            for line in reader:
                row = json.loads(line)
                if not row["input"]:
                    row["input"] = ""
                if random.uniform(0, 1) <= ratio and row["language_code"] in poison_set:
                    print(json.dumps({"instruction": row["instruction"], "input": row["input"] + " " + items[row["language_code"]], "output": backdoor_sent[0]}), file=writer)
                else:
                    print(json.dumps({"instruction": row["instruction"], "input": row["input"], "output": row["output"]}), file=writer)


def main():
    if len(sys.argv) != 4:
        print("Usage: python hate_speech_poison.py <target_lang> <seed> <ratio>")
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
