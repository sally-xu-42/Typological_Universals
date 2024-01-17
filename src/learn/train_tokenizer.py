import argparse
import os
import json

from tokenizers import AddedToken
from tokenizers import Tokenizer
from tokenizers import pre_tokenizers, Regex, normalizers

from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import AutoTokenizer


sample = {
    "en": "Hello, y'all! How are you 😁? (just testing the tokenizer)",
    "ja": "おやすみなさい",
    "it": "Stiamo cercando una gioielleria."
}


def train_tokenizer(model, dataset, lang):

    if model == "gpt2":
        bpe_tokenizer = ByteLevelBPETokenizer()

        files = [f"./data/{dataset}-txt/{lang}_{split}.txt" for split in ["test", "train", "validation"]] # this is always run from root
        
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

    elif model == "ltgbert":
        # adopted from ltgbert paper and repo
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[PAR]"]
        word_piece_tokenizer = WordPiece(unk_token="[UNK]")
        trainer = WordPieceTrainer(
            vocab_size=16384,
            min_frequency=2,
            special_tokens=special_tokens,
            continuing_subword_prefix='@@'
        )

        tokenizer = Tokenizer(word_piece_tokenizer)
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.Nmt(),
            normalizers.NFKC(),
            normalizers.Replace(Regex(" {2,}"), " "),
            normalizers.BertNormalizer(
                lowercase=False,  # args.lowercase,
                clean_text=True,
                handle_chinese_chars=True,
                strip_accents=True
            )
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Split(
                Regex("▁*(\[UNK\]|\[unk\]|n't|N'T|'ll|'LL|'re|'RE|'ve|'VE|'m|'M|'s|'S|'d|'D)(?!\w+)"),
                behavior="isolated",
                invert=False
            ),
            pre_tokenizers.Split(
                Regex("▁*(\.+|#+|\[UNK\]$|\[unk\]$|n't$|N'T$|'ll$|'LL$|'re$|'RE$|'ve$|'VE$|'m$|'M$|'s$|\
                    'S$|'d$|'D$|\d|[^\w▁]){1}"),
                behavior="isolated",
                invert=False
            )
        ])

        files = [f"./data/{dataset}-txt/{lang}_{split}.txt" for split in ["test", "train", "validation"]]

        tokenizer.train(files, trainer)

        tokenizer_path = f'./data/tokenizer/{dataset}-{lang}/{model}_tokenizer'
        tokenizer_file = f'./data/tokenizer/{dataset}-{lang}/{model}_tokenizer/tokenizer.json'
        if not os.path.exists(tokenizer_path):
            os.makedirs(tokenizer_path)

        # save the tokenizer.json file of the trained wordpiece tokenizer
        tokenizer.save(tokenizer_file)
        # tokenizer.model.save(tokenizer_path)

        # model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, tokenizer_type=model)
        model_tokenizer = Tokenizer.from_file(tokenizer_file)
        model_tokenizer.model_max_length = 512

        print(f'Tokenizer vocab size: {model_tokenizer.get_vocab_size()} \n')
        print(f'Tokenizer max sequence length: {model_tokenizer.model_max_length} \n')

        # save the full model tokenizer configuration to a json file
        model_tokenizer.save(f'./data/tokenizer/{dataset}-{lang}/{model}_tokenizer/tokenizer.json')
        output = model_tokenizer.encode(sample[lang])
        print(output.tokens, '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="the tokenizer configuration to use")
    parser.add_argument("dataset", type=str, help="the dataset to be processed")
    parser.add_argument("language", type=str, help="the language of the files")
    args = parser.parse_args()

    print(f'\nTraining a custom {args.model} tokenizer for data {args.dataset}_{args.language}\n')
    train_tokenizer(args.model, args.dataset, args.language)