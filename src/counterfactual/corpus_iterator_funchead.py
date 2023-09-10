# Wrapper around corpus_iterator.py
# Applies a change to the dependency annotation to reverse the head-dep
#   direction for certain relations: cc, case, mark, conj...
# Original Author: Michael Hahn
# Adapted by: Tianyang Xu

from corpus_iterator import CorpusIterator
import sys
import json


def reverse_content_head(sentence, validate=True, CH_CONVERSION_ORDER=["cc","case","cop","mark"], SPECIAL_CC=False):
    """Apply dependency parse convention change (deviation from vanilla UD)

    Args:
        sentence (List[Dict[str,int]]): a list of dictionaries, each corresponding to a word,
        with the UD header names as dictionary keys

    Returns:
        List[Dict[str,int]]: same format as input
    """
    if SPECIAL_CC: # SPECIAL_CC take care of the reverse VO case when we need to differentiate cc connecting verbs or non-verbs
        for i in range(len(sentence)):
            if sentence[i]["dep"] == "cc" or sentence[i]["dep"].startswith("cc:"):
                # lift if head is verb
                head = sentence[i]["head"] - 1
                if sentence[head]["posUni"] == "VERB":
                    grandp = sentence[head]["head"] - 1
                    assert head > -1
                    sentence[i]["head"] = grandp + 1
                    sentence[head]["head"] = i + 1

                    sentence[i]["dep"] = sentence[head]["dep"]
                    sentence[head]["dep"] = "lifted_cc"
                    assert sentence[i]["index"] == i + 1
                    # print("cc position is {} and text is {}".format(i+1, sentence[i]["text"]))
                    # print("now the parent is {} and grandp is {}".format(sentence[i]["head"], sentence[head]["head"]))
                else:
                    continue
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
        CH_CONVERSION_ORDER=["cc","case","cop","mark"], SPECIAL_CC=False
    ):
        self.basis = CorpusIterator(
            filename, language, partition=partition, storeMorph=storeMorph,
        )
        self.validate = validate
        self.CH_CONVERSION_ORDER = CH_CONVERSION_ORDER
        self.SPECIAL_CC = SPECIAL_CC

    def iterator(self, rejectShortSentences=False):
        iterator = self.basis.iterator(rejectShortSentences=rejectShortSentences)
        for sentence, newdoc in iterator:
            r = reverse_content_head(sentence, validate=self.validate, 
                                     CH_CONVERSION_ORDER=self.CH_CONVERSION_ORDER,
                                     SPECIAL_CC=self.SPECIAL_CC)
            if r is None:
                continue
            yield sentence, newdoc

    def getSentence(self, index):
        sentence, newdoc = self.basis.getSentence(index)
        return reverse_content_head(sentence, validate=self.validate, CH_CONVERSION_ORDER=self.CH_CONVERSION_ORDER, SPECIAL_CC=self.SPECIAL_CC), newdoc
