# Wrapper around corpus_iterator.py
# Applies a change to the dependency annotation to reverse the head-dep
#   direction for certain relations: cc, case, mark, conj...
# Original Author: Michael Hahn
# Adapted by: Tianyang Xu

from corpus_iterator import CorpusIterator
import sys
import json

VERB_POS = ["VERB", "AUX"]
NP_POS = ["NOUN", "PROPN", "NUM", "PRON"]


def check_conj(i, p_idx, grandp_idx, sentence):
    """ deal the case of conjucted verbs like 'love and hate'
    input:
    i: sentence[i] for conj "and"
    p_idx: sentence[idx] for later verb like 'hate'
    grandp_idx: sentence[idx] for early verb like 'love'

    goal:
    love -conj-> and -cc-> hate
    love -obj-> her
    """
    sentence[i]["head"] = grandp_idx + 1
    sentence[p_idx]["head"] = i + 1

    sentence[i]["dep"] = sentence[p_idx]["dep"]
    sentence[p_idx]["dep"] = "lifted_cc" # minor fix, deprel => dep
    assert sentence[i]["index"] == i + 1

    sentence[grandp_idx]["has_conj"] = p_idx
    sentence[p_idx]["conj"] = grandp_idx

    return sentence

def children_check(i, ud_label, sentence):
    """ helper for conjugation without cc, now generalized to check if children has a deprel
    i: word idx with deprel as "conj"
    """
    for word in sentence:
        if word["head"] == i+1 and word["dep"] == ud_label:
            return False
    return True

def reverse_content_head(sentence, validate=True, 
                         CH_CONVERSION_ORDER=["cc","case","cop","mark"], 
                         SPECIAL_CC=False, SPECIAL_COP=False):
    """Apply dependency parse convention change (deviation from vanilla UD)

    Args:
        sentence (List[Dict[str,int]]): a list of dictionaries, each corresponding to a word,
        with the UD header names as dictionary keys

    Returns:
        List[Dict[str,int]]: same format as input
    """
    if SPECIAL_COP:
        # new lift convention for copula
        for i in range(len(sentence)):
            if sentence[i]["dep"] == "cop" or sentence[i]["dep"].startswith("cop:"):
                # original: Sally is the head of both I and am
                # goal: I <- am -> Sally
                head = sentence[i]["head"] - 1
                grandp = sentence[head]["head"] - 1
                sentence[i]["head"] = grandp + 1
                sentence[head]["head"] = i + 1

                sentence[i]["dep"] = sentence[head]["dep"]
                sentence[head]["dep"] = "lifted_cop"
                assert sentence[i]["index"] == i + 1

                # move subject cc, and conj in predicate's children to be child of cop
                # move mark in predicate's children to be child of cop
                # move aux in predicate's children to be child of cop
                # move advmod in predicate's children to be child of cop
                for j in range(len(sentence)):
                    if sentence[j]["head"] == head + 1 and j != i and sentence[j]["dep"] in ["nsubj", "csubj", "cc", "conj", "mark", "aux", "advmod"]:
                        sentence[j]["head"] = i + 1

    if SPECIAL_CC: # SPECIAL_CC take care when we need to differentiate cc connecting verbs or non-verbs
        for i in range(len(sentence)):
            if sentence[i]["dep"] == "cc" or sentence[i]["dep"].startswith("cc:"):
                # lift if head is verb
                head = sentence[i]["head"] - 1
                if sentence[head]["posUni"] in VERB_POS or NP_POS:
                    grandp = sentence[head]["head"] - 1
                    assert head > -1
                    sentence = check_conj(i, head, grandp, sentence)
                else:
                    continue

        for i in range(len(sentence)): # deal with conj without cc
            if sentence[i]["dep"] == "conj" and children_check(i, "lifted_cc", sentence):
                # 没有and,只有conj,两边都没有被标记
                p_idx = sentence[i]["head"] - 1
                # 这里其实不用查有没有conj,为了语法对称可以留下。conj在没有and的情况下不可能被标记
                if not sentence[p_idx].get("has_conj", None):
                    sentence[p_idx]["has_conj"] = i
                if not sentence[i].get("conj", None):
                    sentence[i]["conj"] = p_idx

    # find paths that should be reverted
    for dep in CH_CONVERSION_ORDER:
        for i in range(len(sentence)):
            if sentence[i]["dep"] == dep or sentence[i]["dep"].startswith(dep + ":"):
                head = sentence[i]["head"] - 1
                grandp = sentence[head]["head"] - 1
                assert head > -1

                # grandp -> head -> i
                # grandp -> i -> head
                sentence[i]["head"] = grandp + 1
                sentence[head]["head"] = i + 1

                sentence[i]["dep"] = sentence[head]["dep"]
                sentence[head]["dep"] = "lifted_" + dep
                assert sentence[i]["index"] == i + 1

    # make sure none of the original dependency relations remain
    for i in range(len(sentence)):
        if sentence[i]["dep"] in CH_CONVERSION_ORDER:
            if validate:
                sys.stderr.write(json.dumps(sentence))
                sys.stderr.write("\n")
            return None

    return sentence


class CorpusIteratorFuncHead:
    def __init__(
        self, filename, language, partition="train", storeMorph=False, validate=True, 
        CH_CONVERSION_ORDER=["cc","case","cop","mark"], SPECIAL_CC=False, SPECIAL_COP=False
    ):
        self.basis = CorpusIterator(
            filename, language, partition=partition, storeMorph=storeMorph,
        )
        self.validate = validate
        self.CH_CONVERSION_ORDER = CH_CONVERSION_ORDER
        self.SPECIAL_CC = SPECIAL_CC
        self.SPECIAL_COP = SPECIAL_COP

    def iterator(self, rejectShortSentences=False):
        iterator = self.basis.iterator(rejectShortSentences=rejectShortSentences)
        for sentence, newdoc in iterator:
            r = reverse_content_head(sentence, validate=self.validate, 
                                     CH_CONVERSION_ORDER=self.CH_CONVERSION_ORDER,
                                     SPECIAL_CC=self.SPECIAL_CC,
                                     SPECIAL_COP=self.SPECIAL_COP)
            if r is None:
                continue
            yield sentence, newdoc

    def getSentence(self, index):
        sentence, newdoc = self.basis.getSentence(index)
        return reverse_content_head(sentence, validate=self.validate, CH_CONVERSION_ORDER=self.CH_CONVERSION_ORDER, SPECIAL_CC=self.SPECIAL_CC, SPECIAL_COP=self.SPECIAL_COP), newdoc
