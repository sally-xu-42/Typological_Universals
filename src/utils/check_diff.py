def compare_text_files(file1, file2, output_file):

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # Compare the lines and find the differences
    differences = [f"{line1.strip()} | {line2.strip()}" for line1, line2 in zip(lines1, lines2) if line1 != line2]

    # Write the differences to the output file
    with open(output_file, 'w') as output:
        output.write('\n'.join(differences))

# Usage:
if __name__ == "__main__":
    file1 = './data/wiki40b-random/regular/en_ov_test.txt'
    file2 = './data/wiki40b-random/regular/en_ov_test_final.txt'
    output_file = './diff_final.txt'
    compare_text_files(file1, file2, output_file)