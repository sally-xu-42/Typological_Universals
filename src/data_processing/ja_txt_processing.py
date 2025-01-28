import numpy as np
import re
import argparse
import os
import MeCab


def remove_parentheses(text):
    # Remove () and （） brackets and their content
    return re.sub(r'[\(（].*?[\)）]', '', text)


def handle_other_brackets(match):
    content = match.group(1)
    if re.search(r'[a-zA-Z0-9]', content):  # If brackets contain non-Japanese characters or numbers
        return '(REMOVE_SENTENCE)'  # Use as a marker for removal
    else:
        return content  # Otherwise keep only the content inside the brackets


def remove_japanese_punctuation(text):
    # List of common Japanese punctuation characters
    japanese_punctuation = "？？?〜！!。、・「」『』（）〔〕【】〈〉《》「」『』【】［］｛｝〝〟〰–—‛“”‘’‹›«»‐‑‒―⁃−"
    return ''.join(char for char in text if char not in japanese_punctuation)


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
                result = remove_japanese_punctuation(result)
                processed_sentences.append(result)
            else:
                continue  # Skip the entire sentence if \0 is in the sentence

    return '。'.join(processed_sentences) + ('。' if len(processed_sentences) != 0 else '')


def process_ja(args):
    mecab = MeCab.Tagger("-Owakati")
    with open(args.input, "r") as f_in, open(args.output, "w") as f_out:
        for text in f_in:
            text = re.sub("\s+", "", text).strip()
            processed_text = process_japanese_text(text)
            if len(processed_text) != 0:
                f_out.write(processed_text)
                f_out.write("\n")

    print(f"All Japanese text in {args.input} has been processed and saved in {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="./data/wiki40b-txt/ja_test.txt",
        help="input txt file",
    )
    parser.add_argument(
        "--output",
        default="../ja_new_baseline/new_processed_ja_test.txt",
        help="output txt file"
    )
    args = parser.parse_args()
    process_ja(args)

