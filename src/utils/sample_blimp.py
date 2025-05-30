import os
import random
import json
import argparse

SAMPLE_NUM = 5


def sample(source_dir, sample_dir):
    """ The function to randomly sample 5 pairs from JSON files in a specified directory """
    os.makedirs(sample_dir, exist_ok=True)

    for filename in os.listdir(source_dir):  # iterate over all json files
        if filename.endswith(".jsonl"):
            file_path = os.path.join(source_dir, filename)

            # Read the lines from the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Randomly sample 5 lines
            sampled_lines = random.sample(lines, SAMPLE_NUM)

            # Convert sampled lines back to JSON objects
            sampled_json_objects = [json.loads(line) for line in sampled_lines]

            # Create a new filename for the sampled file
            sample_file_path = os.path.join(sample_dir, f'{filename}')

            # Save the sampled JSON objects to a new file
            with open(sample_file_path, 'w') as outfile:
                for obj in sampled_json_objects:
                    json.dump(obj, outfile)
                    outfile.write('\n')

            print(f"Sampled {SAMPLE_NUM} pairs from {filename} to {sample_file_path}")


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
                sentence_good = json_object.get('sentence_good', '')
                sentence_bad = json_object.get('sentence_bad', '')

                # Write the sentences to the output text file, separated by a newline, with an empty line between pairs
                txt_file.write(sentence_good + '\n')
                txt_file.write(sentence_bad + '\n\n')

        print(f"Extracted sentence pairs to {output_txt}")
    except FileNotFoundError:
        print(f"Error: The file {source_json} was not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        help="path to directory where BLiMP is stored",
        default="../BLiMP_data",
    )
    parser.add_argument(
        "--sample_dir",
        help="path to directory where sentences for BLiMP evaluation should be stored",
        default="../sample_BLiMP",
    )
    args = parser.parse_args()

    print(f"Sampling BLiMP json files...\n")

    # sample(args.dir, args.sample_dir)

    all_json_path = os.path.join(args.sample_dir, "all_samples.json")
    all_txt_path = os.path.join(args.sample_dir, "all_samples.txt")

    combine_jsonl_to_json(args.sample_dir, all_json_path)
    extract_sentence_pairs(all_json_path, all_txt_path)
