import stanza
import os
import sys
import argparse
import numpy as np
import pandas as pd

from tatsuki_utils import *

# nlp = load_japanese_nlp()

def check_conjunction(child_id, parent_id, sentence):
    """ mind the difference with check_conj(), this function is for labelling conjunction words
        check_conj() is for omitting conjunction words that should be omitted
    input:
    child_id: sentence[i] for later np head like 'TV' in 'a TV'. after flattened, sentence[i] also could be "and"
    parent_id: sentence[idx] for prev np head like 'light' in 'the light'

    goal:
    light -conj-> and -cc-> TV
    love -obj-> her
    """
    if sentence[child_id - 1]["coarse_dep"] == "conj" and \
            (not sentence[parent_id - 1].get("has_conj", None)) and \
            (not sentence[child_id - 1].get("conj", None)):
        # 展平以后: child不可能有conj标签,因为child是and。parent才会起作用
        # 不展平的情况（没有cc）: child可能有conj标签，parent也可能有has_conj标签，所以还是要check，不需要修改
        # 同一个词有multiple conjugates的情况，且其中一个会被省略：前后都check就可以解决。
        # this conj has been forgotten, should be treated as part of the large np chunk
        print(f"{sentence[child_id - 1]['word']} is not added to {sentence[parent_id - 1]['word']}'s chunk")
        return True
    else:
        # Later np does not belong to this chunk, like in "turn on the light and off the TV"
        return False


def get_noun_descendant(id, sentence):
    """ DFS function for getting all descendants of noun in <NOUN, G>, with depth tracking. """
    stack = [(id, 0)]  # Store node and depth as a tuple
    res = []

    while stack:
        node, depth = stack.pop()  # Unpack the node and its depth
        if node == 0 or not sentence[node - 1].get("children", None):
            continue
        for c in sentence[node - 1]['children']:
            # Check for conjunction and advcl only at the first layer (depth 0)
            # TATSUKI: including nominalization particle to be a part of the noun
            if depth == 0 and (
                    sentence[c - 1]["coarse_dep"] not in ["nummod", "compound", "appos", "flat"] and sentence[c - 1][
                "posUni"] not in ["SCONJ"]):
                continue
            stack.append((c, depth + 1))  # Increment depth for recursive calls
            res.append(c)

    return set([id] + res)


def sanity_dates(sentence):
    # Assuming 'sentence' is a list of dicts with 'text' and 'dep' keys, and 'head' being an index.
    for i, word in enumerate(sentence):
        if word['word'] in months:
            # Check if the previous word is a number and if the head of current word (month) is the number.
            if i > 0 and sentence[i - 1]['word'].isdigit() and word['head'] == i:
                print("Stanza has made an error in dates")
                # Assign the day's dependency to the month (making the month the head)
                word["coarse_dep"] = sentence[i - 1]["coarse_dep"]
                sentence[i - 1]['coarse_dep'] = 'nummod'  # or whatever the correct relation should be
                word["head"] = sentence[i - 1]['head']
                sentence[i - 1]['head'] = i + 1  # Assuming 'head' is 1-indexed and pointing to the current month.
                # Optionally, update the month's head if necessary, e.g., pointing to the year or 'ROOT'.
    return sentence


def swap_order(noun_idx, g_idx, sentence, result, printPair=False):
    """ Helper function for swapping """
    # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
    res = []

    noun_chunk = get_noun_descendant(noun_idx + 1, sentence)
    g_chunk = get_all_descendant(g_idx + 1, sentence)
    # noun_chunk = [i for i in (noun_chunk - g_chunk) if (i <= g_idx + 1)]
    noun_chunk = [i for i in (noun_chunk - g_chunk) if (i >= g_idx + 1)]

    NOUN_POS = [result.index(i) for i in noun_chunk]
    G_POS = [result.index(i) for i in g_chunk]
    MAX_POS = max(max(NOUN_POS), max(G_POS))

    if printPair:
        noun_words = "".join([sentence[i - 1]["word"] for i in noun_chunk])
        g_words = "".join([sentence[i - 1]["word"] for i in g_chunk])
        print("<{}, {}>".format(g_words, noun_words))

    for pos, idx in enumerate(result):
        if pos in G_POS or pos > MAX_POS:
            continue
        res.append(idx)
    res.extend([idx for pos, idx in enumerate(result) if pos in G_POS])
    res.extend([idx for pos, idx in enumerate(result) if pos > MAX_POS])

    return res


def swap(sentence, root, printPair=False):
    """ DFS function for swapping NOUN and G """
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
            for c in sentence[node - 1]["children"]:
                # TATSUKI: avoid 形容詞語幹, i.e., の「よう」な
                # TATSUKI: consider nominalization particle 「の」; e.g., 「すごいの」 should be treated as noun
                if ("名詞" in sentence[node - 1]["posFine"] or [t for t in sentence if
                                                                t["head"] == node and t["posUni"] == "SCONJ" and t[
                                                                    "word"] == "の"]) and sentence[c - 1][
                    'coarse_dep'] in NOUN_G_ARCS:
                    noun_idx, g_idx = node - 1, c - 1
                    marker = False
                    # indicator for <NOUN, G> in English is "of"
                    for i in sentence[g_idx].get("children", []):
                        if sentence[i - 1]['word'] in ["が", "の", "つ"]:  # japanese
                            marker = True

                    # if noun_idx < g_idx and marker: # <Noun, Genitive>
                    if g_idx < noun_idx and marker:  # <Noun, Genitive>
                        num_swap += 1
                        result = swap_order(noun_idx, g_idx, sentence, result, printPair)
                if c not in visited:
                    stack.append(c)

    return num_swap, idx_to_sent(result, sentence)


def test(test_sent, printPair=False):
    doc = nlp(test_sent)
    sentence = doc.sentences[0].to_dict()
    # print(sentence)
    sentence = reverse_content_head(sentence)
    # print(sentence)
    sentence = sanity_dates(sentence)
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
    sentence = reverse_content_head(sentence)
    sentence = sanity_dates(sentence)
    sent = copy.deepcopy(sentence)
    root, sentence = get_all_children(sent, SPECIAL_ADVCL=False, SPECIAL_EXPL=False, SPECIAL_MARK=False)
    num_swaps, ordered = swap(sentence, root, printPair=True)
    dl = get_dl(ordered, sentence, pairs=pairs)
    if pairs:
        dl_original, dl = get_dl(ordered, sentence, pairs=pairs)
        return dl_original, dl
    return dl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", help="conllu file to swap", default="./test_data/ja_NOUN_G_original_sample.conllu"
    )
    parser.add_argument(
        "--output", help="output swapped file", default="./test_data/ja_noun_g_changed_sample.txt"
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
        swap_document(test, input_path="./ja_sample.txt", output_path="./test.txt", printPair=False)
    else:
        corpusIterator = iterator(args.input)
        with open(args.output, "w") as file:
            for i, (sentence, newdoc) in enumerate(corpusIterator):
                sentence = reverse_content_head(sentence)
                sent = copy.deepcopy(sentence)
                root, sent = get_all_children(sent, SPECIAL_ADVCL=False, SPECIAL_EXPL=False, SPECIAL_MARK=False)
                num_swap, output = swap(sent, root, printPair=True)
                if newdoc and i != 0:
                    file.write("\n")
                file.write(output)
                file.write("。")
