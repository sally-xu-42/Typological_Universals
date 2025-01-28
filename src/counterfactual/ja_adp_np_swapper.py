import stanza
import os
import sys
import argparse
import numpy as np
import pandas as pd

from tatsuki_utils import *


# nlp = load_japanese_nlp()

def swap_order(adp_idx, np_idx, sentence, result, printPair=False):
    """ Helper function for swapping """
    # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
    res = []

    if sentence[np_idx]["coarse_dep"] == "nummod":
        # Sanity check: Stanza error for wrong head of "case" in dates.
        # Example: 16 February 2016
        month_idx = sentence[np_idx]["head"] - 1
        if sentence[month_idx]["word"] in months:
            print("Stanza has made an error in dates.")
            np_idx = month_idx

    adp_chunk = get_all_descendant(adp_idx + 1, sentence)
    np_chunk = get_all_descendant(np_idx + 1, sentence)
    np_chunk = set([x for x in (np_chunk - adp_chunk) if x <= np_idx + 1])

    if printPair:
        # print(f"The adposition is {sentence[adp_idx]['text']} and NP head is {sentence[np_idx]['text']}")
        adp_words = "".join([sentence[i - 1]["word"] for i in adp_chunk])
        np_words = "".join([sentence[i - 1]["word"] for i in np_chunk])
        print("<{}, {}>".format(adp_words, np_words))

    ADP_POS = [result.index(i) for i in adp_chunk]
    NP_POS = [result.index(i) for i in np_chunk]
    MAX_POS = len(sentence) - min(min(ADP_POS), min(NP_POS))

    for pos, idx in enumerate(result[::-1]):
        if idx in adp_chunk or pos > MAX_POS - 1: continue
        res.append(idx)
    res.extend([idx for idx in result if idx in adp_chunk])
    res.extend([idx for pos, idx in enumerate(result[::-1]) if pos > MAX_POS - 1])

    res = list(reversed(res))

    return res


def swap(sentence, root, printPair=False, upsample=False):
    """ DFS function for swapping ADP and NP"""
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
            non_last_word_flag = node < len(sentence) - 1

            # TATSUKI: keep the relative order of multiple adpositions
            for c in sentence[node - 1]["children"][::-1]:
                # TATSUKI: consider the nominalization with "の" (連体化助詞 https://www.unixuser.org/~euske/doc/postag/)
                if ("名詞" in sentence[node - 1]["posFine"] or [t for t in sentence if
                                                                t["head"] == node and t["posUni"] == "SCONJ" and t[
                                                                    "word"] == "の"]) and \
                        sentence[c - 1]['coarse_dep'] in ADP_NP_ARCS and \
                        sentence[c - 1]['posUni'] in ADP_POS:
                    np_idx, adp_idx = node - 1, c - 1
                    if np_idx < adp_idx:  # postposition
                        num_swap += 1
                        result = swap_order(adp_idx, np_idx, sentence, result, printPair)
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
        "--input", help="conllu file to swap", default="./test_data/ja_ADP_NP_original_sample.conllu"
    )
    parser.add_argument(
        "--output", help="output swapped file", default="./test_data/ja_adp_np_changed_sample.txt"
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
    #     data = []
    #
    #     for i, (sentence, newdoc) in enumerate(corpusIterator):
    #         dl_original, dl_changed = dl(sentence, pairs=True)
    #         data.append({
    #             'sentence_id': i,
    #             'pair': "ADP_NP",
    #             'lang': "ja",
    #             'dl_original': dl_original,
    #             'dl_changed': dl_changed,
    #             'difference': abs(dl_original - dl_changed)
    #         })
    #
    #     # Create DataFrame and save to CSV
    #     df = pd.DataFrame(data)
    #     output_file = '../dependency_length.csv'  # you can modify the path/name
    #     df.to_csv(output_file, index=False)
    #
    #     # Still print the average difference if needed
    #     sys.stdout.write(f"The average difference of DL between minimal pairs is: {df['difference'].mean()}\n")
    #     sys.stdout.write(f"Data saved to {output_file}\n")
    #     quit()

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