import stanza
import os
import sys
import argparse
import numpy as np
import pandas as pd

from tatsuki_utils import *

# nlp = load_japanese_nlp()

def swap_order(aux_idx, verb_idx, sentence, result, printPair=False):
    """ Helper function for swapping """
    # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
    res = []
    aux_chunk = get_all_descendant(aux_idx + 1, sentence)
    # TATSUKI: auxilary particles will be a part of a word (i.e., at the morphological level), so we flipped a local morpheme order inside verb, not considering VP
    # verb_chunk = get_all_descendant(verb_idx + 1, sentence)
    verb_chunk = [verb_idx + 1]

    # verb_chunk = set([x for x in verb_chunk if (x > aux_idx + 1)])
    verb_chunk = set([x for x in verb_chunk if (x < aux_idx + 1)])
    aux_chunk = set([x for x in (aux_chunk - verb_chunk) if (x > aux_idx)])

    AUX_POS = [result.index(i) for i in aux_chunk]
    VERB_POS = [result.index(i) for i in verb_chunk]
    MIN_POS = len(sentence) - min(min(AUX_POS), min(VERB_POS))

    if printPair:
        aux_words = "".join([sentence[i - 1]["word"] for i in aux_chunk])
        verb_words = "".join([sentence[i - 1]["word"] for i in verb_chunk])
        print("<{}, {}>".format(verb_words, aux_words))

    if len(aux_chunk) == 0: return result

    for pos, idx in enumerate(result[::-1]):
        if idx in aux_chunk or pos > MIN_POS - 1:
            continue
        res.append(idx)

    res.extend([idx for idx in result if idx in aux_chunk])
    res.extend([idx for pos, idx in enumerate(result[::-1]) if pos > MIN_POS - 1])

    res = list(reversed(res))

    return res


def swap(sentence, root, printPair=False):
    """ DFS function for swapping AUX and VERB """
    result = [i for i in range(1, len(sentence) + 1)]
    stack = [root]
    visited = set()
    num_swap = 0

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            if not sentence[node - 1].get("children", None):
                continue
            # TATSUKI: keep relative distance of multiple aux
            for c in sentence[node - 1]["children"][::-1]:
                if sentence[c - 1]['coarse_dep'] in AUX_VERB_ARCS:
                    verb_idx, aux_idx = node - 1, c - 1
                    # subj = sentence[node-1].get('nsubj', 0)
                    # subj_idx = subj - 1
                    # if aux_idx < verb_idx and subj_idx < aux_idx: # <subj, aux, verb>
                    if verb_idx < aux_idx:  # <subj, verb, aux>
                        num_swap += 1
                        result = swap_order(aux_idx, verb_idx, sentence, result, printPair)
                if c not in visited:
                    stack.append(c)

    return num_swap, idx_to_sent(result, sentence)


def test(test_sent, printPair=False):
    doc = nlp(test_sent)
    sentence = doc.sentences[0].to_dict()
    # print(sentence)
    sentence = reverse_content_head(sentence)
    # print(sentence)
    sent = copy.deepcopy(sentence)
    root, sent = get_all_children(sent, SPECIAL_ADVCL=False, SPECIAL_EXPL=False, SPECIAL_MARK=False)
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
    root, sentence = get_all_children(sentence)
    num_swaps, ordered = swap(sentence, root, printPair=True)
    dl = get_dl(ordered, sentence, pairs=pairs)
    if pairs:
        dl_original, dl = get_dl(ordered, sentence, pairs=pairs)
        return dl_original, dl
    return dl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", help="conllu file to swap", default="./test_data/ja_AUX_V_original_sample.conllu"
    )
    parser.add_argument(
        "--output", help="output swapped file", default="./test_data/ja_aux_v_swapped.txt"
    )
    parser.add_argument(
        "--avg_dl_diff",
        help="evaluating average DL difference between minimal pairs",
        action="store_true"
    )
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    if args.avg_dl_diff:
        corpusIterator = iterator(args.input)
        dep_lens_original, dep_lens_changed = [], []

        for i, (sentence, newdoc) in enumerate(corpusIterator):
            dl_original, dl_changed = dl(sentence, pairs=True)
            dep_lens_original.append(dl_original)
            dep_lens_changed.append(dl_changed)

        print(len(dep_lens_changed))
        if len(dep_lens_changed) == len(dep_lens_original):
            differences = [abs(a - b) for a, b in zip(dep_lens_changed, dep_lens_original)]
            # average_difference = sum(differences) / len(differences)
            sys.stdout.write(f"The average difference of DL between minimal pairs is: {np.mean(differences)}\n")
        else:
            sys.stdout.write(f"Wrong file input")

        quit()

    if args.test_run:
        swap_document(test, input_path="./ja_sample.txt", output_path="./test.txt", printPair=False)
    else:
        corpusIterator = iterator(args.input)
        with open(args.output, "w") as file:
            for i, (sentence, newdoc) in enumerate(corpusIterator):
                sentence = reverse_content_head(sentence)
                sent = copy.deepcopy(sentence)
                root, sent = get_all_children(sent, SPECIAL_ADVCL=False, SPECIAL_EXPL=False, SPECIAL_MARK=False)
                num_swap, output = swap(sent, root, printPair=False)
                if newdoc and i != 0:
                    file.write("\n")
                file.write(output)
                file.write("。")