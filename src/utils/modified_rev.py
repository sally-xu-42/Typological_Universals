import argparse
from modified_corpus_iterator_funchead import CorpusIteratorFuncHead
from modified_swapper import Swapper, VOSwapper, ADP_NP_Swapper, COP_PRED_Swapper, AUX_V_Swapper, NOUN_G_Swapper, COMP_S_Swapper, \
        NOUN_RELCL_Swapper

VO_LANG = ["en"]
OV_LANG = ["ja", "zh-cn", "ko"]
NON_SPACE_LANG = ["ja", "ko", "zh-cn"]
REV_PAIR_ORDERS = ["VO", "ADP_NP", "COP_PRED", "AUX_V", "NOUN_G", "COMP_S", "NOUN_RELCL"]

def create_swapper(pair, order, space, upsample):
    """ factory function for instantiating the actual Swapper object """
    if pair == "VO":
        return VOSwapper(order, space, upsample)
    elif pair == "ADP_NP":
        return ADP_NP_Swapper(order, space, upsample)
    elif pair == "COP_PRED":
        return COP_PRED_Swapper(order, space, upsample)
    elif pair == "AUX_V":
        return AUX_V_Swapper(order, space, upsample)
    elif pair == "NOUN_G":
        return NOUN_G_Swapper(order, space, upsample)
    elif pair == "COMP_S":
        return COMP_S_Swapper(order, space, upsample)
    elif pair == "NOUN_RELCL":
        return NOUN_RELCL_Swapper(order, space, upsample)
    else:
        return Swapper(order, space, upsample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", help="code of language, e.g. en, ja, zh-cn", default="en"
    )
    parser.add_argument(
        "--pair",
        help="<X,Y> pair that needs to be reversed, e.g. VO, ADP_NP",
        default="NOUN_G",
    )
    parser.add_argument(
        "--filename",
        help="filename of CONLLU data",
        default="./data/wiki40b-random/regular/en_sample.conllu",
    )
    parser.add_argument(
        "--output",
        help="output file of plain reordered text that has a specific <X,Y> pair reordered",
        default="./data/wiki40b-random/regular/en_s_comp.txt"
    )
    parser.add_argument(
        "--upsample",
        help="upsampling mode to only keep swapped sentences",
        action="store_true"
    )
    parser.add_argument(
        "--upsample_output",
        help="output file of plain reordered text for upsampling",
        default="./src/utils/en_g_noun.txt"
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
        args.filename, args.language, "train", validate=False, CH_CONVERSION_ORDER=[], SPECIAL_CC=True, SPECIAL_COP=True
    )
    corpusIterator = corpus.iterator()

    # don't connect by space in Chinese, Japanese and Korean
    space = False if args.language in NON_SPACE_LANG else True
    # 1 for VO, 2 for OV
    order = 1 if args.language in VO_LANG else 2
    upsample = args.upsample

    # load the swapper for a specific pair
    swapper = create_swapper(args.pair, order, space, upsample)

    if args.upsample:
        # UPSAMPLE MODE for 20 swapped examples
        SAMPLE_NUM = 20
        count = 0
        with open(args.upsample_output, "w") as upsample_file:
            for i, (sentence, newdoc) in enumerate(corpusIterator):
                output = swapper.pipeline(sentence)
                if output:
                    if count != 0:
                        upsample_file.write("\n")
                    upsample_file.write(output)
                    upsample_file.write(". ")  # Add a period after every sentence
                    count += 1
                if count >= SAMPLE_NUM:
                    break
    else:
        # NORMAL MODE
        # iterate over all sentences in a corpus
        with open(args.output, "w") as file:
            for i, (sentence, newdoc) in enumerate(corpusIterator):
                output = swapper.pipeline(sentence)
                if newdoc and i != 0:
                    file.write("\n")
                file.write(output)
                file.write(". ")  # add a period after every sentence, removed the space before period
