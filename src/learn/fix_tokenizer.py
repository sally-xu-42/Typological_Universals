# taken from https://github.com/iconstantinescu/lm-critical-period/blob/main/src/learn/fix_tokenizer.py
# modify bug in huggingface GPT2 tokenizer training
import argparse
import json
import os


def fix_tokenizer(path):

    if os.path.exists(f'{path}/added_tokens.json'):
        f = open(f'{path}/added_tokens.json')
        added_tokens = json.load(f)

        os.remove(f'{path}/added_tokens.json')
        print('Removed added_tokens.json')

        with open(f'{path}/vocab.json', 'r', encoding="utf-8") as f:
            vocab = json.load(f)

            for (key, value) in added_tokens.items():
                vocab.update({key:value})
                print(f'Added {key}:{value} to the vocabulary')

            out = open(f'{path}/vocab.json', 'w')
            json.dump(vocab, out)

        with open(f'{path}/config.json', 'r', encoding="utf-8") as f:
            config = json.load(f)

            config['eos_token_id'] = added_tokens["<|endoftext|>"]
            config['bos_token_id'] = added_tokens["<|endoftext|>"]
            config['pad_token_id'] = added_tokens["<pad>"]

            out = open(f'{path}/config.json', 'w')
            json.dump(config, out)
            print(f'Changed model config')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="the path to the tokenizer to be fixed")
    args = parser.parse_args()

    fix_tokenizer(args.path)