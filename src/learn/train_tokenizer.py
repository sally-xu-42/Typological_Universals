import argparse
import os

from tokenizers import AddedToken
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import AutoTokenizer

sample = {
    "en": "Hello, y'all! How are you 😁? (just testing the tokenizer)",
    "ja": "おやすみなさい",
    "it": "Stiamo cercando una gioielleria.",
    "ja_en": "おやすみなさい"
             "Hello, y'all! How are you 😁? (just testing the tokenizer)",
    "it_en": "Stiamo cercando una gioielleria."
             "Hello, y'all! How are you 😁? (just testing the tokenizer)",
}


def train_tokenizer(model, dataset, lang):

    bpe_tokenizer = ByteLevelBPETokenizer()

    files = [f"./data/{dataset}-txt/{lang}.{split}" for split in ["test", "train", "validation"]] # this is always run from root

    bpe_tokenizer.train(files=files, vocab_size=32000, min_frequency=2)

    tokenizer_path = f'./data/tokenizer/{dataset}-{lang}/{model}_tokenizer'
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)

    # save the vocab.json and merges.txt files of the trained bpe tokenizer
    bpe_tokenizer.save_model(tokenizer_path)

    model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, tokenizer_type=model)
    model_tokenizer.model_max_length = 512
    model_tokenizer.add_special_tokens({"pad_token": AddedToken("<pad>", normalized=True)})

    print(f'Tokenizer vocab size: {len(model_tokenizer)}')
    print(f'Tokenizer max sequence length: {model_tokenizer.model_max_length} \n')

    # save the full model tokenizer configuration files
    model_tokenizer.save_pretrained(tokenizer_path)

    output = model_tokenizer.encode_plus(sample[lang])
    print(output.tokens(), '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="the tokenizer configuration to use")
    parser.add_argument("dataset", type=str, help="the dataset to be processed")
    parser.add_argument("language", type=str, help="the language of the files")
    args = parser.parse_args()

    print(f'\nTraining a custom {args.model} tokenizer for data {args.dataset}_{args.language}\n')
    train_tokenizer(args.model, args.dataset, args.language)