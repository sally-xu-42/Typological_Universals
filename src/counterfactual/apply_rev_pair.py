import argparse
import string
from corpus_iterator_funchead import CorpusIteratorFuncHead
from swapper import Swapper

NON_SPACE_LANG = ["ja", "ko", "zh-cn"]
VO_LANG = ["en"]
OV_LANG = ["ja", "zh-cn", "ko"]
REV_PAIR_ORDERS = ["VO", "ADP_NP", "COP_PRED", "AUX_V", "NOUN_G", "COMP_S"]

OBJ_ARCS = ["ccomp", "lifted_cop", "expl", "iobj", "obj", "obl", "xcomp"]
ADP_NP_ARCS = ["case"]
COP_ARCS = ["lifted_cop"] # cop has been lifted in our design
AUX_VERB_ARCS = ["aux"]
NOUN_G_ARCS = ["nmod"]
NOUN_RELCL_ARCS = ["acl"]
COMP_S_ARCS = ["mark"]

VERB_POS = ["VERB", "AUX"]
ADP_POS = ["ADP"]
NP_POS = ["NOUN", "PRON"]


def makeCoarse(x):
    if ":" in x:
        return x[: x.index(":")]
    return x

def check_mark(child_id, parent_id, sentence):
    # helper function for VO swapper to check if the child is a mark of the parent 
    if sentence[parent_id-1]["posUni"] == "VERB" and \
        sentence[child_id-1]["deprel"] == "mark" and \
        sentence[parent_id-1].get("advcl", False) and \
        sentence[child_id-1]["word"] != "to":
        return True
    else:
        return False


def get_all_children(sentence, SPECIAL_EXPL=True, SPECIAL_ADVCL=True):
    """ Coarsify all the dependent relations, remove all puncts, track all children """
    for line in sentence:
        # make the dependency relation label coarse (ignore stuff after colon)
        line["coarse_dep"] = makeCoarse(line["dep"])

        # identify the root, and skip to next word
        if line["coarse_dep"] == "root":
            root = line["index"]
            continue

        if line["coarse_dep"].startswith("punct"):
            continue

        headIndex = line["head"] - 1
        if line["coarse_dep"] == "nsubj":
            if SPECIAL_EXPL and \
                sentence[headIndex].get("expl", float("inf")) < line["index"]:
                    # change the other nsubj into obj
                    line["coarse_dep"] = "obj"
            else:
                sentence[headIndex]["nsubj"] = line["index"]
        
        if SPECIAL_EXPL and \
            line["coarse_dep"] == "expl" and \
            sentence[headIndex]["posUni"] == "VERB" and \
            line["index"] < line["head"]: # There is...
                sentence[headIndex]["expl"] = line["index"]
                sentence[headIndex]["nsubj"] = line["index"]

        if SPECIAL_ADVCL and \
            line["coarse_dep"] == "advcl" and \
            (line.get("nsubj", None) is None): # I slept while eating
                line["coarse_dep"] = "obj"
                # serves to check for the special case, we add an indicator of whether it's changed from advcl
                # this line contains a verb if parser is correct.
                # we only check if advcl is True when swapping. if so we split the "mark" children with exception of "to"
                if line["upos"] == "VERB": line["advcl"] = True

        sentence[headIndex]["children"] = sentence[headIndex].get("children", []) + [line["index"]]
    return root, sentence


def get_all_descendant(id, sentence, SPECIAL_MARK=False):
    """ DFS function for getting all descendants """
    stack = [id]
    res = []

    while stack:
      node = stack.pop()
      if (node == 0) or (not sentence[node-1].get("children", None)): # no subj associated case, node=0
          continue
      for c in sentence[node-1]['children']:
            if SPECIAL_MARK:
                if check_mark(c, node, sentence):
                    res.append(c)
                    continue
            else:        
                stack.append(c)
                res.append(c)
    return set([id] + res)


def swap_order_V_O(verb_idx, obj_idx, subj, sentence, result, printPair=False):
    """ Helper function for swapping """
    # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
    res = []
    verb_chunk = get_all_descendant(verb_idx + 1, sentence, SPECIAL_MARK=True)
    obj_chunk = get_all_descendant(obj_idx + 1, sentence)
    subj_chunk = get_all_descendant(subj, sentence) # for human names like Arthur Ford
    verb_chunk = set([x for x in (verb_chunk - obj_chunk) if x > max(subj_chunk)])
    
    if printPair:
        v_words = [sentence[i-1]["word"] for i in verb_chunk]
        obj_words = [sentence[i-1]["word"] for i in obj_chunk]
        print("<{}, {}>".format(v_words, obj_words))
    
    if len(verb_chunk) == 0: return result # There is a boy on the farm
    
    VERB_POS = [result.index(i) for i in verb_chunk]
    OBJ_POS = [result.index(i) for i in obj_chunk]
    MAX_POS = max(max(VERB_POS), max(OBJ_POS))

    for pos, idx in enumerate(result):
        if idx in verb_chunk or pos > MAX_POS: continue
        res.append(idx)
    res.extend([idx for idx in result if idx in verb_chunk])
    res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])

    return res


def swap_order_ADP_NP(adp_idx, np_idx, sentence, result, printPair=False):
    """ Helper function for swapping """
    # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
    res = []
    adp_chunk = get_all_descendant(adp_idx + 1, sentence)
    np_chunk = get_all_descendant(np_idx + 1, sentence)
    np_chunk = set([x for x in (np_chunk - adp_chunk) if x > max(adp_chunk)])

    ADP_POS = [result.index(i) for i in adp_chunk]
    NP_POS = [result.index(i) for i in np_chunk]
    MAX_POS = max(max(ADP_POS), max(NP_POS))
    
    if printPair:
        adp_words = [sentence[i-1]["word"] for i in adp_chunk]
        np_words = [sentence[i-1]["word"] for i in np_chunk]
        print("<{}, {}>".format(adp_words, np_words))

    for pos, idx in enumerate(result):
        if idx in adp_chunk or pos > MAX_POS: continue
        res.append(idx)
    res.extend([idx for idx in result if idx in adp_chunk])
    res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])

    return res


def idx_to_sent(idx, sentence, space=True):
    # iterate over all sentences in corpus and write its reversed version
    word_list = [x["word"] for x in sentence]
    if space:
        output = " ".join([word_list[i-1] for i in idx if not (word_list[i-1] in string.punctuation)])
    else:
        output = "".join([word_list[i-1] for i in idx if not (word_list[i-1] in string.punctuation)])
    return output
            

def adp_np_swap(sentence, root, printPair=False):
    """ DFS function for swapping adposition and noun phrase """
    result = [i for i in range(1, len(sentence) + 1)]
    stack = [root]
    visited = set()

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            if not sentence[node-1].get("children", None):
                continue
            for c in sentence[node-1]["children"]:
                if sentence[node-1]['posUni'] in NP_POS and \
                    sentence[c-1]['coarse_dep'] in ADP_NP_ARCS and \
                    sentence[c-1]['posUni'] in ADP_POS:
                        np_idx, adp_idx = node - 1, c - 1
                        if adp_idx < np_idx: # preposition
                            result = swap_order_ADP_NP(adp_idx, np_idx, sentence, result, printPair)
                if c not in visited:
                    stack.append(c)

    return result


def v_o_swap(sentence, root, printPair=False):
    """ DFS function for swapping verb and object"""
    result = [i for i in range(1, len(sentence) + 1)]
    stack = [root]
    visited = set()

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            if not sentence[node-1].get("children", None):
                continue
            for c in sentence[node-1]["children"]:
                if sentence[node-1]['posUni'] in VERB_POS and sentence[c-1]['coarse_dep'] in OBJ_ARCS:
                    verb_idx, obj_idx = node - 1, c - 1
                    subj = sentence[node-1].get('nsubj', 0)
                    subj_idx = subj - 1
                    if obj_idx > subj_idx and obj_idx > verb_idx and verb_idx > subj_idx: # AVOID SPECIAL POSITION
                        result = swap_order_V_O(verb_idx, obj_idx, subj, sentence, result, printPair)
                if c not in visited:
                    stack.append(c)

    return result


SWAP_FUNCTIONS = {"VO": v_o_swap, "ADP_NP": adp_np_swap}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", help="code of language, e.g. en, ja, zh-cn", default="en"
    )
    parser.add_argument(
        "--pair",
        help="<X,Y> pair that needs to be reversed, e.g. VO, ADP_NP",
        default="VO",
    )
    parser.add_argument(
        "--filename",
        help="filename of CONLLU data",
        default="./parse/en_tiny_Stanza.conllu",
    )
    parser.add_argument(
        "--output",
        help="output file of plain reordered text that has a specific <X,Y> pair reordered",
        default="./en_ov.txt"
    )
    args = parser.parse_args()

    # check language
    if args.language not in ["en", "ja"]:
        raise ValueError(f"Specified language is invalid: {args.language}")
    
    # check the pair
    if not args.pair in REV_PAIR_ORDERS: 
        raise ValueError(f"Specified pair is invalid: {args.pair}")

    # load and iterate over a corpus
    corpus = CorpusIteratorFuncHead(
        args.filename, args.language, "train", validate=False, CH_CONVERSION_ORDER=["cop"], SPECIAL_CC=True
    )
    corpusIterator = corpus.iterator()

    # find corresponding function
    swap_pair = SWAP_FUNCTIONS[args.pair]
    # don't connect by space in Chinese, Japanese and Korean
    space = False if args.language in NON_SPACE_LANG else True
    # 1 for VO, 2 for OV
    order = 1 if args.language in VO_LANG else 2

    # iterate over all sentences in a corpus
    with open(args.output, "w") as file:
        for i, (sentence, newdoc) in enumerate(corpusIterator):
            root, sentence = get_all_children(sentence)
            ordered = swap_pair(sentence, root, printPair=False)
            output = idx_to_sent(ordered, sentence, space)
            # Add a new line if the just-processed sentence starts a new document
            # TODO: how about new paragraph?
            if newdoc and i != 0:
                file.write("\n")
            file.write(output)
            file.write(". ")  # add a period after every sentence, removed the space before period