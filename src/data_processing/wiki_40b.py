# Generate text files for Wiki-40b dataset
# Author: Thomas Hikaru Clark (thclark@mit.edu)
# Example Usage:
# python wiki_40b.py
#   --data_dir path/to/save --output_prefix path/to/output --num_train_tokens 20000000 \
#   --num_test_tokens 1000000 --lang_code_list en,fr,es

# Run before running experiments to obtain data. Run from the root directory using shell script

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import re
import argparse
import os
import MeCab

r1 = "_START_ARTICLE_\n[^_]*"
r2 = "_START_PARAGRAPH_\n"
r3 = "_START_SECTION_\n[^_]*"
r4 = "_NEWLINE_"

REGEX = re.compile(f"({r1}|{r2}|{r3}|{r4})")
LANG_CODE_LIST = ["en", "ja"]


def remove_parentheses(text):
    # Remove () and （） brackets and their content
    return re.sub(r'[\(（].*?[\)）]', '', text)


def handle_other_brackets(match):
    content = match.group(1)
    if re.search(r'[a-zA-Z0-9]', content):  # If brackets contain non-Japanese characters or numbers
        return '(REMOVE_SENTENCE)'  # Use as a marker for removal
    else:
        return content  # Otherwise keep only the content inside the brackets


def process_japanese_text(text):
    # Step 1: Remove () and （） brackets and their content
    text = remove_parentheses(text)

    # Step 2: Handle other brackets
    # Define all Japanese bracket types
    ja_brackets = "「」『』〔〕【】〈〉《》「」『』【】［］｛｝〝〟‛""''‹›«»"
    opening_brackets = "".join(set(ja_brackets[::2]))
    closing_brackets = "".join(set(ja_brackets[1::2]))
    pattern = f'[{re.escape(opening_brackets)}](.+?)[{re.escape(closing_brackets)}]'

    text = re.sub("[？?！!]", "。", text)
    sentences = text.split('。')
    processed_sentences = []

    for sentence in sentences:
        if sentence.strip():  # Skip empty sentences
            result = re.sub(pattern, handle_other_brackets, sentence)
            if '(REMOVE_SENTENCE)' not in result:
                # Remove any remaining empty brackets
                result = re.sub(f'[{re.escape(opening_brackets)}]\s*[{re.escape(closing_brackets)}]', '', result)
                processed_sentences.append(result)
            else:
                continue  # Skip the entire sentence if \0 is in the sentence

    return '。'.join(processed_sentences) + ('。' if len(processed_sentences) != 0 else '')


def process_tf_dataset(ds, num_tokens, output_file, lang_code):
    """
    Throwing away redundant special symbols, writing txt file from a tensorflow dataset ds
    """

    token_count = 0
    # print(num_tokens)
    if lang_code == "ja":
        mecab = MeCab.Tagger("-Owakati")

    with open(output_file, "a") as f:
        # Turn to a numpy df so that we can easily extract text
        # numpy_items = tfds.as_numpy(ds)
        for batch in ds.as_numpy_iterator():
            for item in batch.get("text"):
                text = item.decode("UTF-8")
                # Create different way of counting number of words for Japanese
                if lang_code == "ja":
                    # text_no_punc = re.sub(r'[^\w\s]', '', text)
                    # token_count += len(text_no_punc)
                    text = re.sub(REGEX, "", text)
                    text = re.sub("\s+", "", text).strip()
                    processed_text = process_japanese_text(text)
                    if len(processed_text) != 0:
                        f.write(processed_text)
                        f.write("\n")
                        segmented_text = mecab.parse(processed_text)
                        token_count += len(segmented_text.split())
                else:
                    text = re.sub(REGEX, " ", text)
                    text = re.sub("\s+", " ", text).strip()
                    f.write(text)
                    f.write("\n")
                    token_count += len(text.split())
                if num_tokens > 0 and token_count > num_tokens:
                    break
            if num_tokens > 0 and token_count > num_tokens:
                break


def process_lang(lang_code, args):
    """
    Loading Wiki-40b for a specified language, writing in three txt files by calling
    process_tf_dataset().
    lang_code: the language code used in Wiki-40b
    args: specifying language list, output directory and batch_size from command line.
    """

    # Construct a tf.data.Dataset using tfds.load()
    ds = tfds.load(
        f"wiki40b/{lang_code}",
        split="train",
        shuffle_files=True,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    process_tf_dataset(ds, args.num_train_tokens, args.output_prefix + lang_code + "_train.txt", lang_code)
    # if args.num_train_tokens > 0:
    #     process_tf_dataset(ds, args.num_train_tokens, args.output_prefix + lang_code + "_train_" + str(args.num_train_tokens) +".txt")
    # else:
    #   process_tf_dataset(ds, args.num_train_tokens, args.output_prefix + lang_code + "_train.txt")

    ds = tfds.load(
        f"wiki40b/{lang_code}",
        split="test",
        shuffle_files=True,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    process_tf_dataset(ds, args.num_test_tokens, args.output_prefix + lang_code + "_test.txt", lang_code)

    ds = tfds.load(
        f"wiki40b/{lang_code}",
        split="validation",
        shuffle_files=True,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    process_tf_dataset(ds, args.num_valid_tokens, args.output_prefix + lang_code + "_validation.txt", lang_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang_code_list",
        help="comma-separated list of language codes to use, e.g. 'en,ja'",
    )
    parser.add_argument(
        "--num_train_tokens",
        type=float,
        default=-1.0,
        help="max number of training examples to sample",
    )
    parser.add_argument(
        "--num_test_tokens",
        type=float,
        default=-1.0,
        help="max number of test examples to sample",
    )
    parser.add_argument(
        "--num_valid_tokens",
        type=float,
        default=-1.0,
        help="max number of validation examples to sample",
    )
    parser.add_argument(
        "--output_prefix",
        default="./data/wiki40b-txt/",
        help="path to output destination for dataset",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--data_dir",
        default="./data/",
        help="path to save data files to"
    )
    args = parser.parse_args()

    # Extract the directory path, make a new one if it doesn't exist
    dirname = os.path.dirname(args.output_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Example command line:
    # python wiki_40b.py --output_prefix path/to/output --num_train_tokens 20000000 --num_test_tokens 1000000 --lang_code_list en,fr,es
    for lang_code in args.lang_code_list.split(","):
        process_lang(lang_code, args)
