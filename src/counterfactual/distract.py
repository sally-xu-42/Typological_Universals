import argparse
import string
import random
from corpus_iterator_funchead import CorpusIteratorFuncHead
from iso_639 import lang_codes

NON_SPACE_LANG = ["ja", "ko", "zh-cn"]
REV_PAIR_ORDERS = ["VO", "ADP_NP"]
OBJ_ARCS = ["obj"]
VERB_POS = ["VERB", "AUX"]
SEED_LIST = [0.1, 0.3, 0.5, 0.7, 0.9]


def makeCoarse(x):
    if ":" in x:
        return x[: x.index(":")]
    return x


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

        sentence[headIndex]["children"] = sentence[headIndex].get("children", []) + [line["index"]]
    return root, sentence


def get_all_descendant(id, sentence):
    """ DFS function for getting all descendants """
    stack = [id]
    res = []

    while stack:
      node = stack.pop()
      if (node == 0) or (not sentence[node-1].get("children", None)): # no subj associated case, node=0
          continue
      for c in sentence[node-1]['children']:
          stack.append(c)
          res.append(c)
    return set([id] + res)


def distract_shuffle(verb_idx, obj_idx, subj, sentence, result, threshold, verbose=False):
    """ Helper function for swapping """
    # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
    res = []
    verb_chunk = get_all_descendant(verb_idx + 1, sentence)
    obj_chunk = get_all_descendant(obj_idx + 1, sentence)
    subj_chunk = get_all_descendant(subj, sentence) # for human names like Arthur Ford
    verb_chunk = set([x for x in (verb_chunk - obj_chunk) if x > max(subj_chunk)])
    
    seed = random.random()
    
    if seed >= threshold:

        if verbose:
            v_words = " ".join([sentence[i-1]["text"] for i in verb_chunk if i <= (verb_idx + 1)])
            obj_words = " ".join([sentence[i-1]["text"] for i in obj_chunk])
            print("<{}, {}>".format(v_words, obj_words))

        if len(verb_chunk) == 0: return result # There is a boy on the farm
        
        VERB_POS = [result.index(i) for i in verb_chunk]
        OBJ_POS = [result.index(i) for i in obj_chunk]
        MAX_POS = max(max(VERB_POS), max(OBJ_POS))
        # print(MAX_POS)

        for pos, idx in enumerate(result):
            if idx in verb_chunk or pos > MAX_POS: continue
            res.append(idx)
        # print(res)
        res.extend([idx for idx in result if idx in verb_chunk])
        # print(res)
        res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])
        return res
    else:
        return result


def idx_to_sent(idx, sentence, space=True):
    # iterate over all sentences in corpus and write its reversed version
    word_list = [x["word"] for x in sentence]
    if space:
        output = " ".join([word_list[i-1] for i in idx if not (word_list[i-1] in string.punctuation)])
    else:
        output = "".join([word_list[i-1] for i in idx if not (word_list[i-1] in string.punctuation)])
    return output
            

def vo2ov_distract(sentence, root, verbose=False):
    """ DFS function for swapping """
    result = [i for i in range(1, len(sentence) + 1)]
    stack = [root]
    visited = set()
    threshold = random.choice(SEED_LIST)

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            if verbose:
                print(node) # print out index of the node being processed
            if not sentence[node-1].get("children", None):
                continue
            for c in sentence[node-1]["children"]:
                if sentence[node-1]['posUni'] in VERB_POS and sentence[c-1]['coarse_dep'] in OBJ_ARCS:
                    verb_idx, obj_idx = node - 1, c - 1
                    subj = sentence[node-1].get('nsubj', 0)
                    subj_idx = subj - 1
                    if obj_idx > subj_idx and obj_idx > verb_idx and verb_idx > subj_idx: # AVOID SPECIAL POSITION
                        result = distract_shuffle(verb_idx, obj_idx, subj, sentence, result, threshold, verbose)
                    #   print(result)
                if c not in visited:
                    stack.append(c)

    return result


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
        default="./data/wiki40b-random/en_distract.conllu",
    )
    parser.add_argument(
        "--output",
        help="output file of plain reordered text that has a specific <X,Y> pair reordered",
        default="./data/wiki40b-random/en_ov_distract.txt"
    )
    args = parser.parse_args()

    # check language
    if not (args.language in lang_codes.keys() or args.language in lang_codes.values()):
        raise ValueError(f"Specified language is invalid: {args.language}")
    lang_codes_inv = {v: k for k, v in lang_codes.items()}
    lang_code = lang_codes_inv[args.language] if args.language in lang_codes.values() else args.language

    # check the pair to reverse
    if not args.pair in REV_PAIR_ORDERS: 
        raise ValueError(f"Specified pair is invalid: {args.pair}")

    # load and iterate over a corpus, VO to OV
    corpus = CorpusIteratorFuncHead(
        args.filename, args.language, "train", validate=False, CH_CONVERSION_ORDER=["cop"], SPECIAL_CC=True
    )
    corpusIterator = corpus.iterator()

    # find corresponding distractor function
    DISTRACT_FUNCTIONS = {"VO": vo2ov_distract, "ADP_NP": None}
    distract_pair = DISTRACT_FUNCTIONS[args.pair]
    # don't connect by space in Chinese, Japanese and Korean
    space = False if lang_code in NON_SPACE_LANG else True

    # iterate over all sentences in a corpus
    with open(args.output, "w") as file:
        for i, (sentence, newdoc) in enumerate(corpusIterator):
            root, sentence = get_all_children(sentence)
            ordered = distract_pair(sentence, root, verbose=False)
            output = idx_to_sent(ordered, sentence, space)
            if i != 0:
                file.write("\n")
            file.write(output)
            file.write(". ")  # add a period after every sentence, removed the space before period