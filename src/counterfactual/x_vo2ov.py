import sys
import json
import copy
import stanza

OBJ_ARCS = ["ccomp", "lifted_cop", "expl", "iobj", "obj", "obl", "xcomp"]
VERB_POS = ["VERB", "AUX"]

def makeCoarse(x):
    if ":" in x:
        return x[: x.index(":")]
    return x

def reverse_content_head(sentence, validate=True):
    """Apply dependency parse convention change (deviation from vanilla UD)

    Args:
        sentence (List[Dict[str,int]]): a list of dictionaries, each corresponding to a word,
        with the UD header names as dictionary keys

    Returns:
        List[Dict[str,int]]: same format as input
    """
    CH_CONVERSION_ORDER = ["cop"]
    # find paths that should be reverted
    for dep in CH_CONVERSION_ORDER:
        for i in range(len(sentence)):
            if sentence[i]["deprel"] == dep or sentence[i]["deprel"].startswith(dep + ":"):
                head = sentence[i]["head"] - 1
                grandp = sentence[head]["head"] - 1
                assert head > -1

                # grandp -> head -> i
                # grandp -> i -> head
                sentence[i]["head"] = grandp + 1
                sentence[head]["head"] = i + 1

                sentence[i]["deprel"] = sentence[head]["deprel"]
                sentence[head]["deprel"] = "lifted_" + dep
                assert sentence[i]["id"] == i + 1

    # make sure none of the original dependency relations remain
    for i in range(len(sentence)):
        if sentence[i]["deprel"] in CH_CONVERSION_ORDER:
            if validate:
                sys.stderr.write(json.dumps(sentence))
                sys.stderr.write("\n")
            return None

    return sentence

def get_all_children(sentence):
    """ Coarsify all the dependent relations, track all children in a sorted fashion """
    for line in sentence:
        # make the dependency relation label coarse (ignore stuff after colon)
        line["coarse_dep"] = makeCoarse(line["deprel"])

        # identify the root, and skip to next word
        if line["coarse_dep"] == "root":
            root = line["id"]
            continue

        if line["coarse_dep"].startswith("punct"):
            continue

        headIndex = line["head"] - 1
        if line["coarse_dep"] == "nsubj":
            sentence[headIndex]["nsubj"] = line["id"]       
        sentence[headIndex]["children"] = sentence[headIndex].get("children", []) + [line["id"]]

    return root


def idx_to_sent(idx, sentence):
    word_list = [x["text"] for x in sentence]
    output = " ".join([word_list[i-1] for i in idx])
    return output


def get_all_descendant(id, sentence):
    """ DFS function for getting all descendants """
    stack = [id]
    res = []
    while stack:
        node = stack.pop()
        if not sentence[node-1].get("children", None):
            continue
        for c in sentence[node-1]['children']:
            stack.append(c)
            res.append(c)
    return set([id] + res)


def swap_order(verb_idx, obj_idx, subj, sentence, result, verbose=False):
    """ Helper function for swapping """
    # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
    res = []
    verb_chunk = get_all_descendant(verb_idx + 1, sentence)
    obj_chunk = get_all_descendant(obj_idx + 1, sentence)
    verb_chunk = set([x for x in (verb_chunk - obj_chunk) if x > subj])

    if verbose:
        print("Verb chunk moving now is {}".format(verb_chunk))
        print("Object chunk moving now is {}".format(obj_chunk))
    
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


def swap(sentence, root, verbose=False):
    """ DFS function for swapping """
    result = [i for i in range(1, len(sentence) + 1)]
    stack = [root]
    visited = set()

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            if verbose:
                print(node) # print out index of the node being processed

            if not sentence[node-1].get("children", None):
                continue
            for c in sentence[node-1]["children"]:
                if sentence[node-1]['upos'] in VERB_POS and sentence[c-1]['coarse_dep'] in OBJ_ARCS:
                    verb_idx, obj_idx = node - 1, c - 1
                    subj = sentence[node-1].get('nsubj', 0)
                    if obj_idx > subj - 1 and obj_idx > verb_idx: # AVOID SPECIAL POSITION
                      result = swap_order(verb_idx, obj_idx, subj, sentence, result)
                      print(result)
                if c not in visited:
                    stack.append(c)
    print(idx_to_sent(result, sentence))
    return idx_to_sent(result, sentence)


def test(test_sent, nlp, verbose=False):
    doc = nlp(test_sent)
    sentence = doc.sentences[0].to_dict()
    sentence = reverse_content_head(sentence)
    sent = copy.deepcopy(sentence)
    root = get_all_children(sent)
    swap(sent, root, verbose)

if __name__ == "__main__":
    test_sent = "Last night, I eat to live, but then I met somebody"
    nlp = stanza.Pipeline("en", processors='tokenize,mwt,pos,lemma,depparse', verbose=False, use_gpu=False)
    doc = nlp(test_sent)
    sentence = doc.sentences[0].to_dict()
    print(sentence)
    test(test_sent, nlp, verbose=True)
    