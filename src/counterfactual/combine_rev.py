import argparse
import string
from corpus_iterator_funchead import CorpusIteratorFuncHead
from combine_swapper import Swapper, VOSwapper, ADP_NP_Swapper, COP_PRED_Swapper, AUX_V_Swapper, NOUN_G_Swapper, COMP_S_Swapper

VO_LANG = ["en"]
OV_LANG = ["ja", "zh-cn", "ko"]
NON_SPACE_LANG = ["ja", "ko", "zh-cn"]
REV_PAIR_ORDERS = ["VO", "ADP_NP", "COP_PRED", "AUX_V", "NOUN_G", "COMP_S"]

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
    else:
        return Swapper(order, space, upsample)

def idx_to_sent(idx, sentence, space=True):
    # moved from class swapper to here for speed increase
    # iterate over all sentences in corpus and write its reversed version
    word_list = [x["word"] for x in sentence]
    if space:
        output = " ".join([word_list[i-1] for i in idx if not (word_list[i-1] in string.punctuation)])
    else:
        output = "".join([word_list[i-1] for i in idx if not (word_list[i-1] in string.punctuation)])
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", help="code of language, e.g. en, ja, zh-cn", default="en"
    )
    parser.add_argument(
        "--pair",
        help="<X,Y> pair (list) that needs to be reversed, separated by comma, e.g. VO, ADP_NP",
        default="VO,ADP_NP",
    )
    parser.add_argument(
        "--filename",
        help="filename of CONLLU data",
        default="./data/wiki40b-random/regular/en_sample.conllu",
    )
    parser.add_argument(
        "--output",
        help="output file of plain reordered text that has a specific <X,Y> pair reordered",
        default="./data/wiki40b-random/regular/en_ov_np_adp.txt"
    )
    parser.add_argument(
        "--upsample",
        help="upsampling mode to only keep swapped sentences",
        action="store_true"
    )
    parser.add_argument(
        "--upsample_output",
        help="output file of plain reordered text for upsampling",
        default="./data/wiki40b-random/upsample/en_s_comp.txt"
    )
    args = parser.parse_args()

    # check language
    if args.language not in ["en", "ja"]:
        raise ValueError(f"Specified language is invalid: {args.language}")
    
    # Split the string into a list of pairs
    pairs_list = args.pair.split(",")
    for pair in pairs_list:
        # check the pair
        if not pair in REV_PAIR_ORDERS: 
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

    # Create instances of swapper classes based on the pairs
    swappers = []
    for pair in pairs_list:
        # load the swapper for a specific pair
        swapper = create_swapper(pair, order, space, upsample)
        swappers.append(swapper)
    num_swappers = len(swappers) # number of swapper pairs
    if num_swappers != len(pairs_list):
        raise ValueError(f"Number of swappers: {num_swappers}")

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
        if num_swappers == 1:
            with open(args.output, "w") as file:
                for i, (sentence, newdoc) in enumerate(corpusIterator):
                    output = swapper.pipeline(sentence, result=None, printPair=False, combined=False)
                    if newdoc and i != 0:
                        file.write("\n")
                    file.write(output)
                    file.write(". ")  # add a period after every sentence, removed the space before period
        else:
            with open(args.output, "w") as file:
                for i, (sentence, newdoc) in enumerate(corpusIterator):
                    order = [i for i in range(1, len(sentence) + 1)] # initialize order
                    for swapper in swappers:
                        order = swapper.combined_pipeline(sentence, result=order, printPair=False)
                    output = idx_to_sent(order, sentence, space)
                    if newdoc and i != 0:
                        file.write("\n")
                    file.write(output)
                    file.write(". ")  # add a period after every sentence, removed the space before period

        # with open(args.output, "w") as file:
        #     for i, (sentence, newdoc) in enumerate(corpusIterator):
        #         if num_swappers == 1: 
        #             output = swapper.pipeline(sentence, result=None, printPair=False, combined=False)
        #         else:
        #             for idx, swapper in enumerate(swappers):
        #                 if idx != num_swappers-1:
        #                     output = swapper.pipeline(sentence, result=None, printPair=False, combined=True)
        #                 else:
        #                     output = swapper.pipeline(sentence, result=output, printPair=False, combined=False)

        #         if newdoc and i != 0:
        #             file.write("\n")
        #         file.write(output)
        #         file.write(". ")  # add a period after every sentence, removed the space before period