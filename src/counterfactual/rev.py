import argparse
from corpus_iterator_funchead import CorpusIteratorFuncHead
from swapper import Swapper

VO_LANG = ["en"]
OV_LANG = ["ja", "zh-cn", "ko"]
NON_SPACE_LANG = ["ja", "ko", "zh-cn"]
REV_PAIR_ORDERS = ["VO", "ADP_NP", "COP_PRED", "AUX_V", "NOUN_G", "COMP_S"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", help="code of language, e.g. en, ja, zh-cn", default="en"
    )
    parser.add_argument(
        "--pair",
        help="<X,Y> pair that needs to be reversed, e.g. VO, ADP_NP",
        default="VO",
    )
    parser.add_argument(
        "--filename",
        help="filename of CONLLU data",
        default="./data/wiki40b-random/regular/en_sample.conllu",
    )
    parser.add_argument(
        "--output",
        help="output file of plain reordered text that has a specific <X,Y> pair reordered",
        default="./data/wiki40b-random/regular/en_ov_test.txt"
    )
    args = parser.parse_args()

    # check language
    if args.language not in ["en", "ja"]:
        raise ValueError(f"Specified language is invalid: {args.language}")
    
    # check the pair
    if not args.pair in REV_PAIR_ORDERS: 
        raise ValueError(f"Specified pair is invalid: {args.pair}")

    # load and iterate over a corpus
    corpus = CorpusIteratorFuncHead(
        args.filename, args.language, "train", validate=False, CH_CONVERSION_ORDER=["cop"], SPECIAL_CC=True
    )
    corpusIterator = corpus.iterator()

    # don't connect by space in Chinese, Japanese and Korean
    space = False if args.language in NON_SPACE_LANG else True
    # 1 for VO, 2 for OV
    order = 1 if args.language in VO_LANG else 2

    # load the swapper for a specific pair
    swapper = Swapper(args.pair, order, space)

    # iterate over all sentences in a corpus
    with open(args.output, "w") as file:
        for i, (sentence, newdoc) in enumerate(corpusIterator):
            output =swapper.pipeline(sentence)
            if newdoc and i != 0:
                file.write("\n")
            file.write(output)
            file.write(". ")  # add a period after every sentence, removed the space before period