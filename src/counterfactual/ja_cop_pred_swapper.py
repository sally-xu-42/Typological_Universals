import stanza
import os
import sys
import argparse
import numpy as np
import pandas as pd

from tatsuki_utils import *

# nlp = load_japanese_nlp()

def swap_order(cop_idx, pred_idx, sentence, result, printPair=True):
    """ Helper function for swapping """
    # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
    res = []
    cop_chunk = get_all_descendant(cop_idx + 1, sentence)
    pred_chunk = get_all_descendant(pred_idx + 1, sentence)
    # pred_chunk = set([pred_idx+1] + [x for x in (pred_chunk - cop_chunk) if x >= pred_idx and sentence[x-1]["coarse_dep"] in ["aux", "mark"]])
    cop_chunk = set([cop_idx + 1] + [x for x in (cop_chunk - pred_chunk) if
                                     x >= cop_idx and sentence[x - 1]["coarse_dep"] in ["aux", "mark"]])

    # print(cop_chunk, pred_chunk)

    if True:
        cop_words = "".join([sentence[i - 1]["word"] for i in cop_chunk])
        pred_words = "".join([sentence[i - 1]["word"] for i in pred_chunk])
        print("<{}, {}>".format(cop_words, pred_words))

    if len(cop_chunk) == 0: return result  # There is a boy on the farm

    COP_POS = [result.index(i) for i in cop_chunk]
    PRED_POS = [result.index(i) for i in pred_chunk]
    MAX_POS = max(max(COP_POS), max(PRED_POS))

    for pos, idx in enumerate(result):
        if idx in pred_chunk or pos > MAX_POS: continue
        res.append(idx)
    res.extend([idx for idx in result if idx in pred_chunk])
    res.extend([idx for pos, idx in enumerate(result) if pos > MAX_POS])

    return res


def swap(sentence, root, printPair=True):
    """ DFS function for swapping """
    result = [i for i in range(1, len(sentence) + 1)]
    stack = [root]
    visited = set()
    num_swap = 0
    ignore_idx = -1
    subj_idx = -1

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            if not sentence[node - 1].get("children", None):
                continue
            cop_children = [c for c in sentence[node - 1]["children"] if sentence[c - 1]['coarse_dep'] in COP_PRED_ARCS]
            subj_children = [c for c in sentence[node - 1]["children"] if
                             sentence[c - 1]['coarse_dep'] in ["nsubj", "csubj"]]
            if len(cop_children) > 1:
                ignore_idx = min(cop_children)
            if len(subj_children) > 0:
                subj_idx = subj_children[-1]
            for c in sentence[node - 1]["children"]:
                # if sentence[c-1]['coarse_dep'] in ["cop", "lifted_cop", "obl"]:
                #     print(f"{sentence[c-1]['text']} is {sentence[c-1]['coarse_dep']}")
                if sentence[c - 1]['coarse_dep'] in COP_PRED_ARCS and c != ignore_idx and subj_idx > -1:
                    pred_idx, cop_idx = c - 1, node - 1
                    if subj_idx < pred_idx and pred_idx < cop_idx:  # AVOID SPECIAL POSITION
                        num_swap += 1
                        result = swap_order(cop_idx, pred_idx, sentence, result, printPair=True)
                if c not in visited:
                    stack.append(c)
        ignore_idx = -1
        subj_idx = -1

    # with open('./test_data/ja_cop_pred_pairs.txt', "a") as f:
    #     f.write("\n")

    return num_swap, idx_to_sent(result, sentence)
    # return num_swap, result


def test(test_sent, printPair=False):
    doc = nlp(test_sent)
    sentence = doc.sentences[0].to_dict()
    sentence = reverse_content_head(sentence, SPECIAL_COP=True)
    sent = copy.deepcopy(sentence)
    root, sent = get_all_children(sent, SPECIAL_ADVCL=True, SPECIAL_EXPL=True, SPECIAL_MARK=True)
    return swap(sent, root, printPair)


def get_dl(idx, sentence, pairs=True):
    """Returns the summed dependency lengths for a sentence.

    Args:
        sentence (list[dict[str,any]]): sentence

    Returns:
        int: total dependency length of sentence
    """
    dl, dl_original = 0, 0
    mapping = {original_pos: pos + 1 for pos, original_pos in enumerate(idx)}

    for i, word in enumerate(sentence):
        if word["head"] == 0 or word["coarse_dep"] == "root":
            continue
        else:
            # print(mapping)
            dl += abs(mapping[word["head"]] - (i + 1))
            dl_original += abs(word["head"] - (i + 1))
    if pairs:
        return dl_original, dl
    else:
        return dl


def dl(sentence, pairs=True):
    # if self.pair == "NOUN_G":
    #     sentence = self.sanity_dates(sentence)
    sentence = reverse_content_head(sentence, SPECIAL_COP=True)
    sent = copy.deepcopy(sentence)
    root, sentence = get_all_children(sent, SPECIAL_ADVCL=True, SPECIAL_EXPL=True, SPECIAL_MARK=True)
    num_swaps, ordered = swap(sentence, root, printPair=True)
    dl = get_dl(ordered, sentence, pairs=pairs)
    if pairs:
        dl_original, dl = get_dl(ordered, sentence, pairs=pairs)
        return dl_original, dl
    return dl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", help="conllu file to swap", default="./test_data/ja_COP_PRED_original_sample.conllu"
    )
    parser.add_argument(
        "--output", help="output swapped file", default="./test_data/ja_COP_PRED_swapped.txt"
    )
    parser.add_argument(
        "--avg_dl_diff",
        help="evaluating average DL difference between minimal pairs",
        action="store_true"
    )
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    # if args.avg_dl_diff:
    #     corpusIterator = iterator(args.input)
    #     dep_lens_original, dep_lens_changed = [], []
    #
    #     for i, (sentence, newdoc) in enumerate(corpusIterator):
    #         dl_original, dl_changed = dl(sentence, pairs=True)
    #         dep_lens_original.append(dl_original)
    #         dep_lens_changed.append(dl_changed)
    #
    #     print(len(dep_lens_changed))
    #     if len(dep_lens_changed) == len(dep_lens_original):
    #         differences = [abs(a - b) for a, b in zip(dep_lens_changed, dep_lens_original)]
    #         # average_difference = sum(differences) / len(differences)
    #         sys.stdout.write(f"The average difference of DL between minimal pairs is: {np.mean(differences)}\n")
    #     else:
    #         sys.stdout.write(f"Wrong file input")
    #
    #     quit()

    if args.test_run:
        swap_document(test, input_path="./ja_cop_pred_sample.txt", output_path="./new_test.txt", printPair=False)
    else:
        corpusIterator = iterator(args.input)
        with open(args.output, "w") as file:
            for i, (sentence, newdoc) in enumerate(corpusIterator):
                sentence = reverse_content_head(sentence, SPECIAL_COP=True)
                sent = copy.deepcopy(sentence)
                root, sent = get_all_children(sent, SPECIAL_ADVCL=True, SPECIAL_EXPL=True, SPECIAL_MARK=True)
                num_swap, output = swap(sent, root, printPair=False)
                if newdoc and i != 0:
                    file.write("\n")
                file.write(output)
                file.write("。")
