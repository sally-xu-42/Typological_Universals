import re
import argparse

def count_sentences_with_contractions(file_path):
    # Define the list of contractions provided
    contractions = [
        "aren't", "can't", "couldn't", "didn't", "doesn't", "don't", "hadn't", 
        "hasn't", "haven't", "he'd", "he'll", "he's", "I'd", "I'll", "I'm", "I've", 
        "isn't", "let's", "mightn't", "mustn't", "shan't", "she'd", "she'll", "she's", 
        "shouldn't", "that's", "there's", "they'd", "they'll", "they're", "they've", 
        "we'd", "we're", "we've", "weren't", "what'll", "what're", "what's", "what've", 
        "where's", "who'd", "who'll", "who're", "who's", "who've", "won't", "wouldn't", 
        "you'd", "you'll", "you're", "you've"
    ]

    # Regular expression for contractions
    # contraction_pattern = r"\b(?:[a-zA-Z]+'[a-zA-Z]+)\b"
    contraction_pattern = re.compile('|'.join([re.escape(contraction) for contraction in contractions]))
    count = 0

    with open(file_path, 'r') as file:
        for line in file:
            if re.search(contraction_pattern, line):
                print(line)
                count += 1
    
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count sentences with contractions in a text file.')
    parser.add_argument('--file_path', help='Path to the text file')
    args = parser.parse_args()

    result = count_sentences_with_contractions(args.file_path)
    print(f"Number of sentences with contractions: {result}")
