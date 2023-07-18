import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
import argparse
import os

r1 = "_START_ARTICLE_\n[^_]*"
r2 = "_START_PARAGRAPH_\n"
r3 = "_START_SECTION_\n[^_]*"
r4 = "_NEWLINE_"

REGEX = re.compile(f"({r1}|{r2}|{r3}|{r4})")

# Step 1: Convert TensorFlow dataset to NumPy arrays
def process_tf_dataset(ds, num_tokens, output_file):
    # Turn to a numpy df so that we can easily extract text
    # numpy_items = tfds.as_numpy(ds)
    token_count = 0

    with open(output_file, "a") as f:
        for batch in ds.as_numpy_iterator():
            for item in batch.get("text"):
                text = item.decode("UTF-8")
                text = re.sub(REGEX, " ", text)
                text = re.sub("\s+", " ", text).strip()
                f.write(text)
                f.write("\n")
                token_count += len(text.split())
                if num_tokens > 0 and token_count > num_tokens:
                    break

def process_lang(lang_code, args):
    # construct a PyTorch Dataset
    ds = tfds.load(
        f"wiki40b/{lang_code}",
        split="train",
        shuffle_files=True,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    pass
