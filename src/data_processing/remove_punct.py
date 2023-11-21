import argparse
import string
import os
import sys
import stanza

MODELS_DICT = {
    'en': 'en',
    'zh-cn': 'zh',
    'fr': 'fr',
    'ja': 'ja',
    'ko': 'ko',
    'tr': 'tr'
}

def sanity_check():
    
    # Define the required version
    required_version = "1.5.1"
    # Get the actual Stanza version
    actual_version = stanza.__version__

    if actual_version != required_version:
        print(f"Error: Stanza version {actual_version} is installed, but version {required_version} is required.")
        raise Exception("Stanza version is not compatible.")
    else:
        print(f"Stanza version {actual_version} is compatible. Proceed with tokenizing.")

def remove_punct_and_lowercase(input_file, output_file, nlp): 

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if (len(line.strip())) == 0:
                    # detect blank, otherwise there will be an "\n" in output file
                    sys.stderr.write("There was a blank line in the input file\n")
                    continue
            
            doc = nlp(line)
            for sentence in doc.sentences:
                # Remove punctuation from each word and then join them into a sentence
                words_without_punct = [word.text.translate(str.maketrans('', '', string.punctuation)) for word in sentence.words if word.text.translate(str.maketrans('', '', string.punctuation))]
                sentence_text = " ".join(words_without_punct)

                # Lowercase the sentence and add a period at the end
                processed_sentence = sentence_text.lower().strip() + ". "
                f_out.write(processed_sentence)
            f_out.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", 
        help="code of language corpus with 3 splits to be processed, e.g. en, ja, zh-cn", 
        default="en"
    )
    parser.add_argument(
        "--data_dir", 
        help="directory of plain txt data with 3 splits to be processed, e.g. en, ja, zh-cn", 
        default="./data/wiki40b-txt"
    )
    parser.add_argument(
        "--output_dir", 
        help="output directory of plain txt data with 3 splits to be processed, e.g. en, ja, zh-cn", 
        default="./data/wiki40b-baseline"
    )
    args = parser.parse_args()

    # check language
    if args.language not in ["en", "ja"]:
        raise ValueError(f"Specified language is invalid: {args.language}")
    
    sanity_check()
    print(f'\nRunning the baseline conversion from {args.data_dir}/{args.language}')

    stanza.download(MODELS_DICT[args.language])
    # Initialize Stanza pipeline for the specified language
    nlp = stanza.Pipeline(lang=args.language, processors='tokenize')
    print(f'Loaded Stanza {args.language} model...')

    # File names
    input_files = [f"{args.language}_train.txt", f"{args.language}_test.txt", f"{args.language}_validation.txt"]
    output_files = [f"processed_{file}" for file in input_files]

    if not os.path.exists(args.output_dir):
        os.system(f"mkdir -p {args.output_dir}")

    # Add prefix
    input_path = [os.path.join(args.data_dir, filename) for filename in input_files]
    output_path = [os.path.join(args.output_dir, filename) for filename in output_files]

    # Process each file
    for in_file, out_file in zip(input_path, output_path):
        remove_punct_and_lowercase(in_file, out_file, nlp)

