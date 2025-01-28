import stanza
import os
import sys
import argparse
import numpy as np
import pandas as pd

from tatsuki_utils import *


# nlp = load_japanese_nlp()

def swap_order(verb_idx, obj_idx, subj, sentence, result, printPair=False):
    """ Helper function for swapping """
    # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
    res = []
    verb_chunk = get_all_descendant(verb_idx + 1, sentence)
    obj_chunk = get_all_descendant(obj_idx + 1, sentence)
    subj_chunk = get_all_descendant(subj, sentence)
    verb_chunk = set([x for x in (verb_chunk - obj_chunk) if x > max(subj_chunk)])

    # check verb chunk to erase its mark head and desc
    # TATSUKI: this significantly collapses the Japanese sentence
    # verb_chunk = verb_chunk - check_mark(verb_idx + 1, sentence)

    if printPair:
        v_words = "".join([sentence[i - 1]["word"] for i in sorted(verb_chunk)])
        obj_words = "".join([sentence[i - 1]["word"] for i in sorted(obj_chunk)])
        print("<{}, {}>".format(obj_words, v_words))
        # with open('./test_data/ja_vo_pairs.txt', "a") as f:
        #     f.write("<{}, {}>\n".format(obj_words, v_words))

    if len(verb_chunk) == 0: return result  # There is a boy on the farm

    VERB_POS = [result.index(i) for i in verb_chunk]
    OBJ_POS = [result.index(i) for i in obj_chunk]
    MAX_POS = max(max(VERB_POS), max(OBJ_POS))

    for pos, idx in enumerate(result):
        if idx in obj_chunk or pos > MAX_POS: continue
        res.append(idx)
    res.extend([idx for idx in result if idx in obj_chunk])
    res.extend([idx for pos, idx in enumerate(result) if pos > MAX_POS])

    return res


def get_case_markers(node_id, sentence):
    """ Get case markers of a node """
    case_markers = []
    for c in sentence[node_id - 1].get("children", []):
        if sentence[c - 1]['coarse_dep'] == "case":
            case_markers.append(sentence[c - 1]['word'])
    return case_markers


def swap(sentence, root, printPair=True):
    """ DFS function for swapping """
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
            for c in sentence[node - 1]["children"][
                     ::-1]:  # TATSUKI: to keep the relative distance between objects, reverse the children order
                # TATSUKI: if the noun is in a サ変 (sahen) category, and it has a nmod/obj with case marker, then the noun should be treated as verb
                # TATSUKI: "と" should be excluded considering parallel phrase
                # TATSUKI: "の" should be excluded not to mixup it with gentive case
                if (sentence[node - 1]['posUni'] in VERB_POS and sentence[c - 1][
                    'coarse_dep'] in OBJ_ARCS + SUBJ_ARCS and not (
                        set(['が', 'は']) & set(get_case_markers(c, sentence)))) or (
                        sentence[node - 1]['posUni'] in NP_POS and is_sahen(sentence[node - 1]["word"]) and
                        sentence[c - 1]['coarse_dep'] in OBJ_ARCS + ["nmod"] and get_case_markers(c, sentence) and not (
                        set(["と", "の", "や", "が", "は"]) & set(get_case_markers(c, sentence)))):
                    verb_idx, obj_idx = node - 1, c - 1
                    subj = sentence[node - 1].get('nsubj', 0)
                    subj_idx = subj - 1
                    num_swap += 1
                    result = swap_order(verb_idx, obj_idx, subj, sentence, result, printPair=True)
                if c not in visited:
                    stack.append(c)

    return num_swap, idx_to_sent(result, sentence)


def test(test_sent, printPair=False):
    doc = nlp(test_sent)
    sentence = doc.sentences[0].to_dict()
    sentence = reverse_content_head(sentence)
    sent = copy.deepcopy(sentence)
    root, sent = get_all_children(sent, SPECIAL_ADVCL=True, SPECIAL_EXPL=True, SPECIAL_MARK=True, SPECIAL_CC=True)
    return swap(sent, root, printPair=False)


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
        "--input", help="conllu file to swap", default="./test_data/ja_VO_original_sample.conllu"
    )
    parser.add_argument(
        "--output", help="output swapped file", default="./test_data/ja_swapped.txt"
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
    #             'pair': "VO",
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
                root, sent = get_all_children(sent, SPECIAL_ADVCL=True, SPECIAL_EXPL=True, SPECIAL_MARK=True, SPECIAL_CC=True)
                num_swaps, output = swap(sent, root, printPair=False)
                if newdoc and i != 0:
                    file.write("\n")
                file.write(output)
                file.write("。")