import os
import random
import json
import argparse

SAMPLE_NUM = 5

def sample(source_dir, sample_dir):
    """ The function to randomly sample 5 pairs from JSON files in a specified directory """
    os.makedirs(sample_dir, exist_ok=True)

    for filename in os.listdir(source_dir): # iterate over all json files
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
            sample_file_path = os.path.join(sample_dir, f'sample_{filename}')

            # Save the sampled JSON objects to a new file
            with open(sample_file_path, 'w') as outfile:
                for obj in sampled_json_objects:
                    json.dump(obj, outfile)
                    outfile.write('\n')

            print(f"Sampled {SAMPLE_NUM} pairs from {filename} to {sample_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        help="path to directory where BLiMP is stored",
        default="../blimp",
    )
    parser.add_argument(
        "--sample_dir",
        help="path to directory where sentences for BLiMP evaluation should be stored",
        default="../sample_blimp",
    )
    args = parser.parse_args()

    print(f"Sampling BLiMP json files...\n")
    sample(args.dir, args.sample_dir)
