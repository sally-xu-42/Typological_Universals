import argparse
import string
import stanza


def sanity_check():
    
    # Define the required version
    required_version = "1.5.1"
    # Get the actual Stanza version
    actual_version = stanza.__version__

    if actual_version != required_version:
        print(f"Error: Stanza version {actual_version} is installed, but version {required_version} is required.")
        raise Exception("Stanza version is not compatible.")
    else:
        print(f"Stanza version {actual_version} is compatible. Proceed with parsing.")

def remove_punct_and_lowercase(input_file, output_file, nlp):

    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
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
    args = parser.parse_args()

    # check language
    if args.language not in ["en", "ja"]:
        raise ValueError(f"Specified language is invalid: {args.language}")
    
    # Initialize Stanza pipeline for the specified language
    nlp = stanza.Pipeline(lang=args.language, processors='tokenize')
    print(f'Loaded Stanza {args.language} model...')

    # File names
    input_files = [f"{args.language}_train.txt", f"{args.language}_test.txt", f"{args.language}_validation.txt"]
    output_files = [f"processed_{file}" for file in input_files]

    # Process each file
    for in_file, out_file in zip(input_files, output_files):
        remove_punct_and_lowercase(in_file, out_file, nlp)

