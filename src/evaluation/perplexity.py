import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import os
import math

device = "cuda"

def calculate_token_perplexity(input_file, model, tokenizer):
    """
    calculating perplexity on the test set
    """

    total_loss = 0.0
    total_sents = 0
    max_seq_length = model.config.max_position_embeddings
    with torch.no_grad():
        with open(input_file, "r") as f:
            for sentence in f:
                sentence = sentence.strip()  # Remove leading/trailing whitespace
                if not sentence:  # skip empty sentence
                    continue
                inputs = tokenizer(sentence, truncation=True, padding=True, max_length=max_seq_length,
                                   return_tensors="pt").to(device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                if math.isnan(loss.item()):
                    print("fuck")
                    continue
                total_loss += loss.item()
                # total_tokens += inputs["input_ids"].size(1)
                total_sents += 1

    print(total_loss)
    print(total_sents)
    if total_sents == 0:
        return float('nan')  # Avoid division by zero
    perplexity = torch.exp(torch.tensor(total_loss / total_sents)).item()

    return perplexity


def calculate_perplexity(model_path, input_path, output_file):
    """
    calculating perplexity on the 2 minimal pairs files
    """

    model_path = os.path.normpath(model_path)
    splits = model_path.split("/")

    if "checkpoint-" in splits[-1]:
        # Extract the checkpoint number if it exists
        checkpoint_name = splits[-1]
        checkpoint_number = checkpoint_name.split('-')[-1]
        model_dir = splits[-2]
    else:
        # If no checkpoint is present, set checkpoint_number to None or handle it accordingly
        checkpoint_number = None
        model_dir = splits[-1]

    results_dict = {}
    keys = ["model", "lang", "pair", "seed", "step"]
    values = model_dir.split('-')
    print(values)
    values.append(checkpoint_number)

    for key, value in zip(keys, values):
        results_dict[key] = value

    model, lang, pair, seed, step = values

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config).to(device)
    model.eval()

    if pair == "NO_PUNCT":
        # Run baseline minimal pairs
        for p in ["VO", "AUX_V", "NOUN_G", "COP_PRED", "ADP_NP"]:
            changed_file = os.path.join(input_path, f"{lang}_{p}_changed.txt")
            same_file = os.path.join(input_path, f"{lang}_{p}_same.txt")
            # Calculate perplexities
            perplexities_changed = calculate_token_perplexity(changed_file, model, tokenizer)
            perplexities_same = calculate_token_perplexity(same_file, model, tokenizer)

            result = []
            result.append(('changed', results_dict["model"], results_dict["lang"], results_dict["pair"] + f'-{p}',
                           results_dict["seed"], results_dict["step"], perplexities_changed))
            result.append(('same', results_dict["model"], results_dict["lang"], results_dict["pair"] + f'-{p}',
                           results_dict["seed"], results_dict["step"], perplexities_same))

            new_df = pd.DataFrame(result, columns=['condition', 'model', 'lang', 'pair', 'seed', 'step', 'perplexity'])

            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                # Concatenate the existing DataFrame with the new DataFrame
                df = pd.concat([existing_df, new_df])
            else:
                df = new_df

            df.to_csv(output_file, index=False)
            print(f"Output saved to {output_file} for baseline minimal pairs")

        return

    # Prepare data
    changed_file = os.path.join(input_path, f"{lang}_{pair}_changed.txt")
    same_file = os.path.join(input_path, f"{lang}_{pair}_same.txt")

    # Calculate perplexities
    perplexities_changed = calculate_token_perplexity(changed_file, model, tokenizer)
    perplexities_same = calculate_token_perplexity(same_file, model, tokenizer)

    result = []
    result.append(('changed', results_dict["model"], results_dict["lang"], results_dict["pair"], results_dict["seed"],
                   results_dict["step"], perplexities_changed))
    result.append(('same', results_dict["model"], results_dict["lang"], results_dict["pair"], results_dict["seed"],
                   results_dict["step"], perplexities_same))

    # Prepare DataFrame
    new_df = pd.DataFrame(result, columns=['condition', 'model', 'lang', 'pair', 'seed', 'step', 'perplexity'])

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        # Concatenate the existing DataFrame with the new DataFrame
        df = pd.concat([existing_df, new_df])
    else:
        df = new_df

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")


def calculate_sentence_surprisal(sentence, model, tokenizer):
    """
    calculating surprisal of a single sentence
    """
    max_seq_length = model.config.max_position_embeddings
    inputs = tokenizer(sentence, truncation=True, max_length=max_seq_length, return_tensors="pt").to(device)
    num_tokens = inputs['input_ids'].size(1)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # outputs = model(input_ids)
        loss = outputs.loss

        loss = loss.item()
        total_loss = loss * num_tokens
        loss_per_char = total_loss / len(sentence)

    return loss_per_char


def calculate_surprisal(model_path, input_path, output_file):
    """
    wrapper function for calculating surprisal on minimal pairs
    """

    model_path = os.path.normpath(model_path)
    splits = model_path.split("/")

    if "checkpoint-" in splits[-1]:
        checkpoint_name = splits[-1]
        checkpoint_number = checkpoint_name.split('-')[-1]
        model_dir = splits[-2]
    else:
        # If no checkpoint is present, set checkpoint_number to None or handle it accordingly
        checkpoint_number = None
        model_dir = splits[-1]

    results_dict = {}
    keys = ["model", "lang", "pair", "seed", "step"]
    values = model_dir.split('-')
    values.append(checkpoint_number)

    for key, value in zip(keys, values):
        results_dict[key] = value
    model, lang, pair, seed, step = values

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config).to(device)
    model.eval()

    # For baseline
    if pair == "NO_PUNCT":
        # Run baseline minimal pairs
        for p in ["VO", "AUX_V", "NOUN_G", "COP_PRED", "ADP_NP"]:
            changed_file = os.path.join(input_path, f"{lang}_{p}_changed.txt")
            original_file = os.path.join(input_path, f"{lang}_{p}_changed_original.txt")

            with open(changed_file, 'r') as f:
                sentences_changed = [line.strip() for line in f.readlines() if line.strip()]

            with open(original_file, 'r') as f:
                sentences_original = [line.strip() for line in f.readlines() if line.strip()]

            # Calculate surprisals
            surprisals_changed = [calculate_sentence_surprisal(sentence, model, tokenizer) for sentence in
                                  sentences_changed]
            surprisals_original = [calculate_sentence_surprisal(sentence, model, tokenizer) for sentence in
                                   sentences_original]

            result = []
            for sentence_id, surprisal in enumerate(surprisals_changed):
                result.append((sentence_id, 'changed', results_dict["model"], results_dict["lang"],
                               results_dict["pair"] + f'-{p}', results_dict["seed"], results_dict["step"], surprisal))
            for sentence_id, surprisal in enumerate(surprisals_original):
                result.append((sentence_id, 'original', results_dict["model"], results_dict["lang"],
                               results_dict["pair"] + f'-{p}', results_dict["seed"], results_dict["step"], surprisal))

            # Prepare DataFrame
            new_df = pd.DataFrame(result, columns=['sentence_id', 'condition', 'model', 'lang', 'pair', 'seed', 'step',
                                                   'surprisal'])

            output_file = os.path.join("./ja_pairs_eval_baseline", f"{pair}-{p}_{seed}_{step}_gpt2.csv")  # testing
            df = new_df

            # Save to CSV
            df.to_csv(output_file, index=False)
            print(f"Output saved to {output_file}")

        return

    # Prepare data
    changed_file = os.path.join(input_path, f"{lang}_{pair}_changed.txt")
    original_file = os.path.join(input_path, f"{lang}_{pair}_changed_original.txt")

    with open(changed_file, 'r') as f:
        sentences_changed = [line.strip() for line in f.readlines() if line.strip()]

    with open(original_file, 'r') as f:
        sentences_original = [line.strip() for line in f.readlines() if line.strip()]

    # Calculate surprisals
    surprisals_changed = [calculate_sentence_surprisal(sentence, model, tokenizer) for sentence in sentences_changed]
    surprisals_original = [calculate_sentence_surprisal(sentence, model, tokenizer) for sentence in sentences_original]

    result = []
    for sentence_id, surprisal in enumerate(surprisals_changed):
        result.append((sentence_id, 'changed', results_dict["model"], results_dict["lang"], results_dict["pair"],
                       results_dict["seed"], results_dict["step"], surprisal))
    for sentence_id, surprisal in enumerate(surprisals_original):
        result.append((sentence_id, 'original', results_dict["model"], results_dict["lang"], results_dict["pair"],
                       results_dict["seed"], results_dict["step"], surprisal))

    # Prepare DataFrame
    new_df = pd.DataFrame(result,
                          columns=['sentence_id', 'condition', 'model', 'lang', 'pair', 'seed', 'step', 'surprisal'])

    output_file = os.path.join("./new_ja_pairs_eval", f"{pair}_{seed}_{step}_gpt2.csv")  # testing
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        df = pd.concat([existing_df, new_df])
    else:
        df = new_df

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate sentence-level surprisal or token-level perplexity using a pretrained model.")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to the input folder containing sentences.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file for scores.")
    parser.add_argument("--token", action="store_true", help="token level evaluation flag")

    args = parser.parse_args()

    if args.token:
        calculate_perplexity(args.model, args.input, args.output)
    else:
        calculate_surprisal(args.model, args.input, args.output)