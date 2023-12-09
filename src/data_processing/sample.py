import random
import argparse
import re
import os
import sys

SAMPLE_NUM = 100

def sample(lang, parse_dir, output_dir):

    if not os.path.exists(parse_dir):
        os.system(f"mkdir -p {parse_dir}")

    input_path = os.path.join(parse_dir, f"{lang}_train.conllu")
    output_text_path = os.path.join(output_dir, f"{lang}_sample.txt")
    output_parse_path = os.path.join(output_dir, f"{lang}_sample.conllu")

    with open(input_path, "r") as f_in:
        sent_counter = 0
        for line in f_in:
            if line.strip() == "":
                sent_counter += 1
    num_sentences = sent_counter - 1
    sys.stderr.write(f"INFO: There are {num_sentences} sentences in this CoNLL-U file\n")

    # Randomly select 100 numbers from the range 1 to num_sentences
    random_idx_list = sorted(random.sample(range(num_sentences), SAMPLE_NUM))

    with open(input_path, "r") as f_in, open(output_parse_path, "w") as f_parse_out, open(output_text_path, "w") as f_text_out:
        current_idx = 0
        for line in f_in:        
            if current_idx in random_idx_list:
                f_parse_out.write(line)
                if line.startswith("# text"):
                    match = re.search(r'# text = (.+)', line)
                    if match: 
                        random_sentence = match.group(1).strip()
                        f_text_out.write(random_sentence)
                        f_text_out.write("\n")
                if line.strip() == "":
                    current_idx += 1
            elif line.strip() == "":
                current_idx += 1
            else:
                continue

    sys.stderr.write(f"INFO: {SAMPLE_NUM} random sentences saved to {output_text_path}\n")
    sys.stderr.write(f"INFO: {SAMPLE_NUM} random parses saved to {output_parse_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang", help="language code such as en, fr, zh-cn, etc.", default="en"
    )
    parser.add_argument(
        "--parse_dir",
        help="path to directory where CONLLU parses of sentences are stored",
        default="./parse",
    )
    parser.add_argument(
        "--sample_dir",
        help="path to directory where sentences for manual evaluation should be stored",
        default="./data/wiki40b-manual",
    )
    args = parser.parse_args()

    print(f"Sampling the parses and sentences for {args.lang}...\n")
    sample(args.lang, args.parse_dir, args.sample_dir)