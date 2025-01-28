import os
import random
import json
import argparse

def combine_jsonl_to_json(source_dir, output_file):
    """ Combine multiple JSONL files into a single JSON file. """
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(source_dir):  # Iterate over all JSONL files
            if filename.endswith(".jsonl"):
                file_path = os.path.join(source_dir, filename)

                with open(file_path, 'r') as file:
                    for line in file:
                        outfile.write(line)

    print(f"Combined JSONL files into {output_file}")


def extract_sentence_pairs(source_json, output_txt):
    """Extract 'sentence_good' and 'sentence_bad' pairs from a combined JSON file and save them to a text file."""
    try:
        with open(source_json, 'r') as json_file, open(output_txt, 'w') as txt_file:
            for line in json_file:
                # Convert each line from the JSON file back to a JSON object
                json_object = json.loads(line)

                # Extract the 'sentence_good' and 'sentence_bad' fields
                sentence_good = json_object.get('good_sentence', '')
                sentence_bad = json_object.get('bad_sentence', '')

                # Write the sentences to the output text file, separated by a newline, with an empty line between pairs
                txt_file.write(sentence_good + '\n')
                txt_file.write(sentence_bad + '\n\n')

        print(f"Extracted sentence pairs to {output_txt}")
    except FileNotFoundError:
        print(f"Error: The file {source_json} was not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_dir",
        help="path to directory where sentences for BLiMP evaluation should be stored",
        default="../counterfactual_JBLiMP",
    )
    args = parser.parse_args()

    print(f"Extracting JBLiMP sentences...\n")

    all_json_path = os.path.join(args.sample_dir, "ja_all_samples.json")
    all_txt_path = os.path.join(args.sample_dir, "ja_all_samples.txt")

    combine_jsonl_to_json(args.sample_dir, all_json_path)
    extract_sentence_pairs(all_json_path, all_txt_path)
