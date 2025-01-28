import stanza
import argparse
import os
import sys

MODELS_DICT = {
    'en': 'en',
    'ja': 'ja'
}


def sanity_check(lang):
    # Define the required version
    required_version = {
        'en': "1.5.1",
        'ja': "1.6.1"
    }
    # Get the actual Stanza version
    actual_version = stanza.__version__

    if actual_version != required_version[lang]:
        print(f"Error: Stanza version {actual_version} is installed, but version {required_version[lang]} is required.")
        raise Exception("Stanza version is not compatible.")
    else:
        print(f"Stanza version {actual_version} is compatible. Proceed with parsing.")


def parse(lang, data_dir, parse_dir, partitions, test_run=False):
    if not os.path.exists(parse_dir):
        os.system(f"mkdir -p {parse_dir}")

    if lang == "ja":
        stanza.download(MODELS_DICT[lang], package="gsdluw")
        # TATSUKI: use long unit word (LUW) model
        nlp = stanza.Pipeline("ja",
                              package="GSDLUW",
                              verbose=False,
                              use_gpu=True,
                              download_method=None)
    else:
        stanza.download(MODELS_DICT[lang])
        nlp = stanza.Pipeline(MODELS_DICT[lang],
                              processors='tokenize,lemma,pos,depparse',
                              verbose=False,
                              use_gpu=True,
                              download_method=None)
    print(f'Loaded Stanza {MODELS_DICT[lang]} model...')

    for partition in partitions.split(","):
        input_path = os.path.join(data_dir, f"{lang}_{partition}.txt")
        if test_run:
            output_path = os.path.join(parse_dir, f"{lang}_tiny.conllu")
        else:
            output_path = os.path.join(parse_dir, f"{lang}_{partition}.conllu")

        with open(input_path) as f_in, open(output_path, "w") as f_out:
            doc_counter = 0

            for line in f_in:
                if (len(line.strip())) == 0:
                    # detect blank, otherwise there will be an "\n" in output file
                    sys.stderr.write("There was a blank line in the input file\n")
                    continue

                doc = nlp(line)
                f_out.write("# newdoc\n")
                f_out.write("{:C}".format(doc))
                f_out.write("\n\n")

                # five lines for test mode
                doc_counter += 1
                if test_run and doc_counter >= 5:
                    sys.stderr.write("Test finished!")
                    exit()

    sys.stderr.write("Job finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang", help="language code such as en, fr, zh-cn, etc.", default="en"
    )
    parser.add_argument(
        "--data_dir",
        help="path to data directory with original (normal-order) text",
        default="./data/wiki40b-txt",
    )
    parser.add_argument(
        "--parse_dir",
        help="path to directory where CONLLU parses of sentences should be stored",
        default="./parse",
    )
    parser.add_argument(
        "--partitions",
        default="train,test,validation",
        help="comma-seprated list of partitions",
    )
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    sanity_check(args.lang)
    print(f'\nParsing the data to CoNLL format from {args.data_dir}/{args.lang}')
    parse(args.lang, args.data_dir, args.parse_dir, args.partitions, args.test_run)
