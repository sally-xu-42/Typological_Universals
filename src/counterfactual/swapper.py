import string
# import colorama

class Swapper():
    """ Class for swapping a pair of specified grammatical elements """

    OBJ_ARCS = ["ccomp", "lifted_cop", "expl", "iobj", "obj", "obl", "xcomp"]
    VERB_POS = ["VERB", "AUX"]

    ADP_NP_ARCS = ["case"]
    ADP_POS = ["ADP"]
    NP_POS = ["NOUN", "PRON"]

    COP_PRED_ARCS = ["lifted_cop"]

    AUX_VERB_ARCS = ["aux"]

    NOUN_G_ARCS = ["nmod"]

    NOUN_RELCL_ARCS = ["acl"]

    COMP_S_ARCS = ["mark"]

    def __init__(self, pair, order=1, space=True):

        self.pair = pair
        self.order = order
        self.space = space
        self.swap_functions = {"VO": self.v_o_swap, 
                               "ADP_NP": self.adp_np_swap,
                               "COP_PRED": None,
                               "AUX_V": None,
                               "NOUN_G": None,
                               "COMP_S": None
                               }
        self.swap_pair = self.swap_functions[pair]

        self.SPECIAL_MARK = True if pair == "vo" else False # leave special case for mark in VO swapping
        self.SPECIAL_EXPL = True if pair == "vo" else False # leave special case for "there is" in VO swapping
        self.SPECIAL_ADVCL = True if pair == "vo" else False # leave special case for "there is" in VO swapping

    def makeCoarse(self, x):
        if ":" in x:
            return x[: x.index(":")]
        return x
    
    def idx_to_sent(self, idx, sentence, space=True):
        # iterate over all sentences in corpus and write its reversed version
        word_list = [x["word"] for x in sentence]
        if space:
            output = " ".join([word_list[i-1] for i in idx if not (word_list[i-1] in string.punctuation)])
        else:
            output = "".join([word_list[i-1] for i in idx if not (word_list[i-1] in string.punctuation)])
        return output
    
    def check_mark(self, child_id, parent_id, sentence):
        """ helper function for VO swapper to check if the child is a mark of the parent """
        if sentence[parent_id-1]["posUni"] == "VERB" and \
            sentence[child_id-1]["deprel"] == "mark" and \
            sentence[parent_id-1].get("advcl", False) and \
            sentence[child_id-1]["word"] != "to":
            return True
        else:
            return False
    
    def get_all_descendant(self, id, sentence):
        """ DFS function for getting all descendants """
        stack = [id]
        res = []

        while stack:
            node = stack.pop()
            if (node == 0) or (not sentence[node-1].get("children", None)):
                continue
            if not self.SPECIAL_MARK:
                for c in sentence[node-1]['children']:
                    stack.append(c)
                    res.append(c)
            else:
                for c in sentence[node-1]['children']:
                    if not self.check_mark(c, node, sentence):
                        stack.append(c)
                        res.append(c)

        return set([id] + res)
    
    def get_all_children(self, sentence):
        """ Coarsify all the dependent relations, remove all puncts, track all children """
        for line in sentence:
            # make the dependency relation label coarse (ignore stuff after colon)
            line["coarse_dep"] = self.makeCoarse(line["dep"])

            # identify the root, and skip to next word
            if line["coarse_dep"] == "root":
                root = line["index"]
                continue

            if line["coarse_dep"].startswith("punct"):
                continue

            headIndex = line["head"] - 1
            if line["coarse_dep"] == "nsubj":
                if self.SPECIAL_EXPL and \
                    sentence[headIndex].get("expl", float("inf")) < line["index"]:
                        # change the other nsubj into obj
                        line["coarse_dep"] = "obj"
                else:
                    sentence[headIndex]["nsubj"] = line["index"]
            
            if self.SPECIAL_EXPL and \
                line["coarse_dep"] == "expl" and \
                sentence[headIndex]["posUni"] == "VERB" and \
                line["index"] < line["head"]: # There is...
                    sentence[headIndex]["expl"] = line["index"]
                    sentence[headIndex]["nsubj"] = line["index"]

            if self.SPECIAL_ADVCL and \
                line["coarse_dep"] == "advcl" and \
                (line.get("nsubj", None) is None): # I slept while eating
                    line["coarse_dep"] = "obj"
                    # serves to check for the special case, we add an indicator of whether it's changed from advcl
                    # this line contains a verb if parser is correct.
                    # we only check if advcl is True when swapping. if so we split the "mark" children with exception of "to"
                    if line["upos"] == "VERB": line["advcl"] = True

            sentence[headIndex]["children"] = sentence[headIndex].get("children", []) + [line["index"]]
        return root, sentence
    
    def swap_order_V_O(self, verb_idx, obj_idx, subj, sentence, result, printPair=False):
        """ Helper function for swapping verb and object"""
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []
        verb_chunk = self.get_all_descendant(verb_idx + 1, sentence)
        obj_chunk = self.get_all_descendant(obj_idx + 1, sentence)
        subj_chunk = self.get_all_descendant(subj, sentence) # for human names like Arthur Ford
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
    
    def v_o_swap(self, sentence, root, printPair=False):
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
                    if sentence[node-1]['posUni'] in Swapper.VERB_POS and \
                    sentence[c-1]['coarse_dep'] in Swapper.OBJ_ARCS:
                        verb_idx, obj_idx = node - 1, c - 1
                        subj = sentence[node-1].get('nsubj', 0)
                        subj_idx = subj - 1
                        if obj_idx > subj_idx and obj_idx > verb_idx and verb_idx > subj_idx: # AVOID SPECIAL POSITION
                            result = self.swap_order_V_O(verb_idx, obj_idx, subj, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return result
    
    def swap_order_ADP_NP(self, adp_idx, np_idx, sentence, result, printPair=False):
        """ Helper function for swapping ADP and NP"""
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []
        adp_chunk = self.get_all_descendant(adp_idx + 1, sentence)
        np_chunk = self.get_all_descendant(np_idx + 1, sentence)
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
    
    def adp_np_swap(self, sentence, root, printPair=False):
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
                    if sentence[node-1]['posUni'] in Swapper.NP_POS and \
                        sentence[c-1]['coarse_dep'] in Swapper.ADP_NP_ARCS and \
                        sentence[c-1]['posUni'] in Swapper.ADP_POS:
                            np_idx, adp_idx = node - 1, c - 1
                            if adp_idx < np_idx: # preposition
                                result = self.swap_order_ADP_NP(adp_idx, np_idx, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return result
    
    def pipeline(self, sentence):
        root, sentence = self.get_all_children(sentence)
        ordered = self.swap_pair(sentence, root, printPair=False)
        output = self.idx_to_sent(ordered, sentence, self.space)
        return output
    

