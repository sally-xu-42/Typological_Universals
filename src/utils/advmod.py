def extract_advmod_words(file_path):
    advmod_words = set()  # Using a set to avoid duplicates

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.startswith('#'):  # Skip comment lines
                parts = line.strip().split('\t')
                if len(parts) > 3 and parts[7] == 'advmod':
                    advmod_words.add(parts[1])  # Add the word to the set

    return advmod_words


if __name__ == "__main__":
    # Replace 'your_file.conllu' with your actual file path
    file_path = './parse/en_train.conllu'
    output_file = './src/utils/advmod.txt'
    advmod_words = extract_advmod_words(file_path)

    # Write the unique words with 'advmod' deprel to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in advmod_words:
            f.write(word + '\n')

    print(f"Words with 'advmod' deprel have been written to {output_file}")

