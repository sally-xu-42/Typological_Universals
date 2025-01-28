import stanza
import copy
import sys
import json

from spacy import displacy
from pathlib import Path

import MeCab
import unidic_lite

tagger = MeCab.Tagger(f"-r /cluster/home/tianyxu/mecabrc -d {unidic_lite.DICDIR}")

HEADER = [
    "index",
    "word",
    "lemma",
    "posUni",
    "posFine",
    "morph",
    "head",
    "dep",
    "_",
    "_",
]

OBJ_ARCS = ["ccomp", "lifted_cop", "expl", "iobj", "obj", "obl", "xcomp"]
SUBJ_ARCS = ["nsubj", "csubj"]
VERB_POS = ["VERB"]
NP_POS = ["NOUN", "PROPN", "NUM", "PRON"]  # added PRON and NUM

ADP_NP_ARCS = ["case"]
ADP_POS = ["ADP"]
COP_PRED_ARCS = ["lifted_cop"]
AUX_VERB_ARCS = ["aux"]
NOUN_G_ARCS = ["nmod"]
NOUN_RELCL_ARCS = ["acl:relcl"]
COMP_S_ARCS = ["mark"]

COARSE_EXCEPTIONS = ["acl:relcl"]

ja_punctuations = "。、・「」『』（）〔〕【】〈〉《》「」『』【】［］｛｝〝〟〰–—‛“”‘’‹›«»‐‑‒―⁃−"
ja_brackets = "「」『』（）〔〕【】〈〉《》「」『』【】［］｛｝〝〟‛“”‘’‹›«»"

copula_adverb = ["である", "でない", "ではない", "じゃない", "らしい", "かもしれない"]
copula_verb = ["居る", "有る", "成る"]

months = [
    "January", "Jan", "january", "jan",
    "February", "Feb", "february", "feb",
    "March", "Mar", "march", "mar",
    "April", "Apr", "april", "apr",
    "May", "may",
    "June", "Jun", "june", "jun",
    "July", "Jul", "july", "jul",
    "August", "Aug", "august", "aug",
    "September", "Sep", "september", "sep", "Sept", "sept", "Sep",
    "October", "Oct", "october", "oct",
    "November", "Nov", "november", "nov",
    "December", "Dec", "december", "dec"
]


def load_japanese_nlp():
    # TATSUKI: use long unit word (LUW) model
    nlp = stanza.Pipeline("ja", package="GSDLUW", verbose=False, use_gpu=True)
    return nlp


def makeCoarse(x):
    # leave exceptions
    if x in COARSE_EXCEPTIONS:
        return x
    if ":" in x:
        return x[: x.index(":")]
    return x


def idx_to_sent(idx, sentence):
    word_list = [x["word"] for x in sentence]
    output = "".join([word_list[i - 1] for i in idx if not (word_list[i - 1] in ja_punctuations)])
    return output


def children_check(i, ud_label, sentence):
    """ helper for conjugation without cc, now generalized to check if children has a deprel
    i: word idx with deprel as "conj"
    """
    for word in sentence:
        if word["head"] == i + 1 and word["coarse_dep"] == ud_label:
            return False
    return True


def is_sahen(word):
    return 'サ変' in tagger.parse(word).split("\t")[4]


def get_all_descendant(id, sentence):
    """ DFS function for getting all descendants, with depth tracking. """
    stack = [(id, 0)]  # Store node and depth as a tuple
    res = []

    while stack:
        node, depth = stack.pop()  # Unpack the node and its depth
        if node == 0 or not sentence[node - 1].get("children", None):
            continue
        for c in sentence[node - 1]['children']:
            stack.append((c, depth + 1))  # Increment depth for recursive calls
            res.append(c)

    return set([id] + res)


def reverse_content_head(sentence, validate=True, SPECIAL_CC=True, SPECIAL_COP=False, SPECIAL_ACL=True):
    """Apply dependency parse convention change (deviation from vanilla UD)

    Args:
        sentence (List[Dict[str,int]]): a list of dictionaries, each corresponding to a word,
        with the UD header names as dictionary keys

    Returns:
        List[Dict[str,int]]: same format as input
    """

    # if SPECIAL_ACL:
    # IndexError
    # # original: A の <- よう な <- B
    # # goal: A のよう な <-B
    # # TATSUKI: 助動詞語幹 https://www.unixuser.org/~euske/doc/postag/
    # # TATSUKI: memo ような、ようである、ようだ interferes with copula
    # for i in range(2,len(sentence)-1):
    #     if sentence[i-1]["word"]=="の" and sentence[i]["word"] == "よう" and sentence[i+1]["posUni"]=="AUX":
    #         child = sentence[i-2]
    #         head = sentence[i]["head"] - 1
    #         if child["dep"] == "nmod" and child["posUni"] in NP_POS:
    #             sentence[i-1]["word"] = "のよう"
    #             sentence[i-1]["lemma"] = "の様"
    #             sentence[i-2]["dep"] = "lifted_" + sentence[i]["dep"]
    #             sentence[i-2]["head"] = head + 1
    #             sentence[i+1]["head"] = child["index"]
    #             sentence = sentence[:i] + sentence[i+1:]
    #             for idx in range(len(sentence)):
    #                 if idx >= i:
    #                     sentence[idx]["index"] -= 1
    #                     sentence[idx]["head"] -= 1
    #                 elif sentence[idx]["head"] >= i:
    #                     sentence[idx]["head"] -= 1

    if SPECIAL_COP:
        for i in range(len(sentence)):
            # TATSUKI: ignore the case of AなB (B being A), which has a special word order
            # TATSUKI: consider the case of Aは...である (copula adverb)
            if (sentence[i]["dep"].startswith("cop") and sentence[i]["word"] != "な") and [t for t in sentence if
                                                                                           t["head"] == sentence[i][
                                                                                               "head"] and t[
                                                                                               "dep"] in SUBJ_ARCS] \
                    or (
                    sentence[i]["lemma"] in copula_adverb and sentence[i]["posUni"] == "AUX" and [t for t in sentence if
                                                                                                  t["head"] ==
                                                                                                  sentence[i][
                                                                                                      "head"] and t[
                                                                                                      "dep"] in SUBJ_ARCS]):
                # original: Sally is the head of both I and am
                # goal: I <- am -> Sally
                head = sentence[i]["head"] - 1  # obj
                grandp = sentence[head]["head"] - 1

                sentence[i]["head"] = grandp + 1
                sentence[head]["head"] = i + 1

                sentence[i]["dep"] = sentence[head]["dep"]
                sentence[head]["dep"] = "lifted_cop"
                assert sentence[i]["index"] == i + 1

                subj_id = min([t["index"] for t in sentence if t["head"] == head + 1 and t["dep"] in SUBJ_ARCS])

                # move subject cc, and conj in predicate's children to be child of cop
                for j in range(len(sentence)):
                    if sentence[j]["head"] == head + 1 and j != i and sentence[j]["dep"] in ["nsubj", "csubj", "cc",
                                                                                             "conj", "mark", "aux",
                                                                                             "advcl"]:  # "advmod" "obl"
                        sentence[j]["head"] = i + 1
                    elif sentence[j]["head"] == head + 1 and j < subj_id - 1:
                        sentence[j]["head"] = i + 1

            # TATSUKI: some verbs correspond to the English word "be", which should be treated as a copula if it has a obl (e.g., 私は公園にいる)
            if sentence[i]["lemma"] in copula_verb and sentence[i]["posUni"] == "VERB":
                for t in sentence:
                    if t["head"] == i + 1 and t["dep"] in ["obl"]:
                        t["dep"] = "lifted_cop"

    if SPECIAL_CC:
        for i in range(len(sentence)):
            if sentence[i]["dep"] == "advcl" and sentence[i]["head"] == i + 2 and sentence[sentence[i]["head"] - 1][
                "posUni"] == "VERB" and [c for c in sentence if
                                         c["head"] == i + 1 and c["dep"] == "mark" and c["word"] == "て" and c[
                                             "posUni"] == "SCONJ"]:
                # lift if head is VERB
                # 在 I love and hate her的case中，这样没有问题因为 love -conj-> and -cc-> hate
                # 确保obj连在第一个verb上
                head = sentence[i]["head"] - 1
                for j in range(len(sentence)):
                    if sentence[j]["head"] == i + 1:
                        sentence[j]["head"] = head + 1
                sentence[i]["dep"] = "conj"

    return sentence


def get_all_children(sentence, SPECIAL_EXPL=True, SPECIAL_ADVCL=True, SPECIAL_MARK=True, SPECIAL_CC=True):
    """ Coarsify all the dependent relations, track all children and subject """

    for line in sentence:
        # CASE: Coarsify, skip root and puncts
        line["coarse_dep"] = makeCoarse(line["dep"])

        if "root" in line["coarse_dep"]:
            root = line["index"]
            continue

        if line["coarse_dep"].startswith("punct"):
            continue

        # TATSUKI: expl is not used in Japanese UD

        # CASE: check for finite clauses
        # leave a special case for "to" in the "advcl/xcomp/ccomp" -> VERB -> "mark" cycle
        headIndex = line["head"] - 1
        if SPECIAL_ADVCL and \
                line["coarse_dep"] == "advcl" and \
                (line.get("nsubj", None) is None) and (
                line.get("case", None) or line.get("aux", None) or line.get("mark",
                                                                            None)):  # avoid compound verb; the targeted word should have any functional word
            # differentiate nonfinite and finite adverbial clause
            line["coarse_dep"] = "obj"
            # serves to check for the special case, we add an indicator of whether it's changed from advcl
            # this line contains a verb if parser is correct.
            # we only check if advcl is True when swapping. if so we split the "mark" children with exception of "to"
            if line["posUni"] == "VERB": line["advcl"] = True

        # extend the special case to "xcomp" and "ccomp"
        if SPECIAL_MARK and \
                line["coarse_dep"] in ["xcomp", "ccomp"] and \
                line["posUni"] == "VERB":
            line["split_mark"] = True

        sentence[headIndex]["children"] = sentence[headIndex].get("children", []) + [line["index"]]
    return root, sentence


import re

brackets = re.compile(r'（.+）')


def swap_document(test, input_path, output_path, printPair, remove_brackets=False, skip_brackets=False):
    with open(input_path, "r") as f1, open(output_path, "w") as f2:
        for line in f1:  # Iterate through each line in the input file
            line = line.replace(" ", "")
            if remove_brackets:
                line = brackets.sub('', line)
            if skip_brackets and [b for b in ja_brackets if b in line]:
                continue
            processed_line = test(line.strip(), printPair)
            f2.write(processed_line + '。\n')


def processSentence(sentence):
    sentence = list(map(lambda x: x.split("\t"), sentence.split("\n")))
    newdoc = False

    result = []
    for i in range(len(sentence)):
        if sentence[i][0].startswith("#"):
            if sentence[i][0].startswith("# newdoc"):
                newdoc = True
            continue
        if "-" in sentence[i][0]:  # if it is NUM-NUM
            continue
        if "." in sentence[i][0] or "。" in sentence[i][0]:
            continue

        # sentence = list of dicts, where each key is a field name (see HEADER)
        sentence[i] = dict([(y, sentence[i][x]) for x, y in enumerate(HEADER)])
        sentence[i]["head"] = int(sentence[i]["head"])
        sentence[i]["index"] = int(sentence[i]["index"])
        sentence[i]["word"] = sentence[i]["word"].lower()
        sentence[i]["dep"] = sentence[i]["dep"].lower()
        result.append(sentence[i])

    return result, newdoc


def iterator(filename):
    with open(filename) as f_in:
        buffer = []
        for line in f_in:
            if line != "\n":
                buffer.append(line)
            else:
                sentence = "".join(buffer).strip()
                buffer = []
                sentence, newdoc = processSentence(sentence)
                yield sentence, newdoc
