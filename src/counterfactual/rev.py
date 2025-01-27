import sys
import argparse
import numpy as np
from corpus_iterator_funchead import CorpusIteratorFuncHead
from swapper import Swapper, BASELINESwapper, VOSwapper, ADP_NP_Swapper, COP_PRED_Swapper, AUX_V_Swapper, \
    NOUN_G_Swapper

VO_LANG = ["en"]
OV_LANG = ["ja"]
NON_SPACE_LANG = ["ja"]
REV_PAIR_ORDERS = ["BASELINE", "VO", "ADP_NP", "COP_PRED", "AUX_V", "NOUN_G"]


def create_swapper(pair, order, space, upsample):
    """ factory function for instantiating the Swapper object """
    if pair == "BASELINE":
        return BASELINESwapper(order, space, upsample)
    elif pair == "VO":
        return VOSwapper(order, space, upsample)
    elif pair == "ADP_NP":
        return ADP_NP_Swapper(order, space, upsample)
    elif pair == "COP_PRED":
        return COP_PRED_Swapper(order, space, upsample)
    elif pair == "AUX_V":
        return AUX_V_Swapper(order, space, upsample)
    elif pair == "NOUN_G":
        return NOUN_G_Swapper(order, space, upsample)
    else:
        return Swapper(order, space, upsample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", help="code of language, e.g. en, ja", default="en"
    )
    parser.add_argument(
        "--pair",
        help="<X,Y> pair that needs to be reversed, e.g. VO, ADP_NP",
        default="VO",
    )
    parser.add_argument(
        "--filename",
        help="filename of CONLLU data",
        default="./parse/en_validation.conllu",
    )
    parser.add_argument(
        "--output",
        help="output file of plain reordered text that has a specific <X,Y> pair reordered",
        default="./en_counterfactual/en_vo_validation.txt"
    )
    parser.add_argument(
        "--upsample",
        help="upsampling mode to only keep swapped sentences",
        action="store_true"
    )
    parser.add_argument(
        "--get_dl_only",
        help="dl mode to calculate dependency length",
        action="store_true"
    )
    parser.add_argument(
        "--avg_dl_diff",
        help="evaluating average DL difference between minimal pairs",
        action="store_true"
    )
    args = parser.parse_args()

    if args.language not in ["en", "ja"]:
        raise ValueError(f"Specified language is invalid: {args.language}")

    if not args.pair in REV_PAIR_ORDERS:
        raise ValueError(f"Specified pair is invalid: {args.pair}")

    corpus = CorpusIteratorFuncHead(
        args.filename,
        args.language,
        CH_CONVERSION_ORDER=[],
        SPECIAL_CC=True,
        SPECIAL_COP=True
    )
    corpusIterator = corpus.iterator()

    # Chinese, Japanese and Korean are not connected by space
    space = False if args.language in NON_SPACE_LANG else True
    # 1 for VO, 2 for OV
    order = 1 if args.language in VO_LANG else 2
    upsample = args.upsample

    # load the swapper
    swapper = create_swapper(args.pair, order, space, upsample=False)

    # NORMAL MODE
    # iterate over all sentences in a corpus
    with open(args.output, "w") as file:
        for i, (sentence, newdoc) in enumerate(corpusIterator):
            _, output = swapper.pipeline(sentence)
            if newdoc and i != 0:
                file.write("\n")
            file.write(output)
            if args.language == "en":
                file.write(". ")  # add a period after every sentence
            else:
                file.write("。")
    # if args.get_dl_only:
    #     dep_lens, sent_lens = [], []
    #     for i, (sentence, newdoc) in enumerate(corpusIterator):
    #         dep_lens.append(swapper.dl(sentence))
    #         sent_lens.append(len(sentence))
    #     sys.stdout.write(f"Avg Sentence Length: {np.mean(sent_lens)}\n")
    #     sys.stdout.write(f"Avg Dependency Length: {np.mean(dep_lens)}\n")
    #     quit()
    #
    # if args.avg_dl_diff:
    #     dep_lens_original, dep_lens_changed = [], []
    #     for i, (sentence, newdoc) in enumerate(corpusIterator):
    #         dl_original, dl_changed = swapper.dl(sentence, pairs=True)
    #         dep_lens_original.append(dl_original)
    #         dep_lens_changed.append(dl_changed)
    #
    #     if len(dep_lens_changed) == len(dep_lens_original):
    #         differences = [abs(a - b) for a, b in zip(dep_lens_changed, dep_lens_original)]
    #         sys.stdout.write(f"The average difference of DL between minimal pairs is: {np.mean(differences)}\n")
    #     else:
    #         sys.stdout.write(f"Wrong file input")
    #
    #     quit()
    #
    # if args.upsample: # test swapping
    #     SAMPLE_NUM = 20
    #     count = 0
    #     with open(args.output, "w") as upsample_file:
    #         for i, (sentence, newdoc) in enumerate(corpusIterator):
    #             output = swapper.pipeline(sentence)
    #             if output:
    #                 if count != 0:
    #                     upsample_file.write("\n")
    #                 upsample_file.write(output)
    #                 upsample_file.write(". ")  # Add a period after every sentence
    #                 count += 1
    #             if count >= SAMPLE_NUM:
    #                 break