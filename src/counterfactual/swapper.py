import string
# import colorama

class Swapper():
    """ Class for swapping a pair of specified grammatical elements """

    OBJ_ARCS = ["ccomp", "lifted_cop", "expl", "iobj", "obj", "obl", "xcomp"]
    VERB_POS = ["VERB", "AUX"]

    ADP_NP_ARCS = ["case"]
    ADP_POS = ["ADP"]
    NP_POS = ["NOUN", "PROPN"]

    COP_PRED_ARCS = ["lifted_cop"]

    AUX_VERB_ARCS = ["aux"]

    NOUN_G_ARCS = ["nmod"]

    NOUN_RELCL_ARCS = ["acl"]

    COMP_S_ARCS = ["mark"]

    def __init__(self, pair, order=1, space=True):

        self.pair = pair
        self.order = order
        self.space = space

        self.SPECIAL_CC = True if pair == "VO" else False
        self.SPECIAL_COP = True if pair == "VO" else False
        self.SPECIAL_MARK = True if pair == "VO" else False # leave special case for mark in VO swapping
        self.SPECIAL_EXPL = True if pair == "VO" else False # leave special case for "there is" in VO swapping
        self.SPECIAL_ADVCL = True if pair == "VO" else False # leave special case for "there is" in VO swapping

    def makeCoarse(self, x):
        if ":" in x:
            return x[: x.index(":")]
        return x
    
    def idx_to_sent(self, idx, sentence, space=True):
        # iterate over all sentences in corpus and write its reversed version
        word_list = [x["word"] for x in sentence]
        # print(idx)
        # print(word_list)
        if space:
            output = " ".join([word_list[i-1] for i in idx if not (word_list[i-1] in string.punctuation)])
        else:
            output = "".join([word_list[i-1] for i in idx if not (word_list[i-1] in string.punctuation)])
        return output
    
    def check_mark(self, child_id, parent_id, sentence):
        """ helper function for VO swapper to check if the child is a mark of the parent """
        if sentence[parent_id-1]["posUni"] == "VERB" and \
            sentence[child_id-1]["coarse_dep"] == "mark" and \
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
                    if line["posUni"] == "VERB": line["advcl"] = True
            
            # TODO: extend the special case to "xcomp" and "ccomp"?
            if self.SPECIAL_MARK and \
            line["coarse_dep"] in ["xcomp", "ccomp"] and \
            line["posUni"] == "VERB":
                line["split_mark"] = True
            
            # change the head of the object if head is a later verb in sentence and has a prior conj verb
            # and the prior conj has no objs
            if self.SPECIAL_CC and \
            line["coarse_dep"] in Swapper.OBJ_ARCS and \
            sentence[headIndex].get("has_conj", None):
                # forget it has a conj if there is an object directly connected to it
                right_verb_idx = sentence[headIndex]["has_conj"]
                sentence[right_verb_idx]["conj"] = None
                sentence[headIndex]["has_conj"] = None

            if self.SPECIAL_CC and \
            line["coarse_dep"] in Swapper.OBJ_ARCS and \
            sentence[headIndex].get("conj", None):
                # change the head verb if the prior conj verb has no obj
                line["head"] = sentence[headIndex]["conj"] + 1
                headIndex = sentence[headIndex]["conj"]

            sentence[headIndex]["children"] = sentence[headIndex].get("children", []) + [line["index"]]
        return root, sentence

    def pipeline(self, sentence):
        root, sentence = self.get_all_children(sentence)
        ordered = self.swap(sentence, root, printPair=False)
        output = self.idx_to_sent(ordered, sentence, self.space)
        return output
        
    def swap_pair(self):
        pass

    def swap(self):
        pass

   
class VOSwapper(Swapper):
    def __init__(self, order=1, space=True):
        super().__init__("VO", order, space)

    def swap_pair(self, verb_idx, obj_idx, subj, sentence, result, printPair=False):
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
         
    def swap(self, sentence, root, printPair=False):
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
                            result = self.swap_pair(verb_idx, obj_idx, subj, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return result


class ADP_NP_Swapper(Swapper):
    def __init__(self, order=1, space=True):
        super().__init__("ADP_NP", order, space)
    
    def swap_pair(self, adp_idx, np_idx, sentence, result, printPair=False):
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
    
    def swap(self, sentence, root, printPair=False):
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
                                result = self.swap_pair(adp_idx, np_idx, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return result


class COP_PRED_Swapper(Swapper):
    def __init__(self, order=1, space=True):
        super().__init__("COP_PRED", order, space)

    def swap_pair(self, cop_idx, pred_idx, sentence, result, printPair=False):
        """ Helper function for swapping """
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []
        cop_chunk = self.get_all_descendant(cop_idx + 1, sentence)
        pred_chunk = self.get_all_descendant(pred_idx + 1, sentence)
        cop_chunk = set([x for x in (cop_chunk - pred_chunk) if x > cop_idx])

        COP_POS = [result.index(i) for i in cop_chunk]
        PRED_POS = [result.index(i) for i in pred_chunk]
        MAX_POS = max(max(COP_POS), max(PRED_POS))

        if printPair:
            cop_words = " ".join([sentence[i-1]["word"] for i in cop_chunk])
            pred_words = " ".join([sentence[i-1]["word"] for i in pred_chunk])
            print("<{}, {}>".format(cop_words, pred_words))
        
        if len(cop_chunk) == 0: return result

        for pos, idx in enumerate(result):
            if idx in cop_chunk or pos > MAX_POS:
                # print(f"skipped {idx}")
                continue
            # print(f"this {idx} is being added")
            res.append(idx)
        # print([idx for idx in result if idx in cop_chunk])
        res.extend([idx for idx in result if idx in cop_chunk])
        # print(res)
        res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])

        return res
    
    def swap(self, sentence, root, printPair=False):
        """ DFS function for swapping copula and predicate"""
        # num_swap = 0
        result = [i for i in range(1, len(sentence) + 1)]
        stack = [root]
        visited = set()

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                if not sentence[node-1].get("children", None):
                    # print(f"{node} has no children")
                    continue
                for c in sentence[node-1]["children"]:
                    if sentence[c-1]['coarse_dep'] in Swapper.COP_PRED_ARCS:
                        cop_idx, pred_idx = node - 1, c - 1
                        if cop_idx < pred_idx: # <cop, pred>
                            # num_swap += 1
                            result = self.swap_pair(cop_idx, pred_idx, sentence, result, printPair)
                            # if verbose:
                            #     print(f"Result at the {num_swap}-th step is: {idx_to_sent(result, sentence)}")
                    if c not in visited:
                        stack.append(c)
        
        # print(f"Total number of swaps is: {num_swap}")
        return result


class AUX_V_Swapper(Swapper):
    def __init__(self, order=1, space=True):
        super().__init__("AUX_V", order, space)
    
    def swap_pair(self, aux_idx, verb_idx, sentence, result, printPair=False):
        """ Helper function for swapping """
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []
        aux_chunk = self.get_all_descendant(aux_idx + 1, sentence)
        # print(aux_chunk)
        verb_chunk = self.get_all_descendant(verb_idx + 1, sentence)
        verb_chunk = set(i for i in verb_chunk if i > aux_idx + 1)
        # print(verb_chunk)
        aux_chunk = set([x for x in (aux_chunk - verb_chunk) if x > aux_idx])

        AUX_POS = [result.index(i) for i in aux_chunk]
        VERB_POS = [result.index(i) for i in verb_chunk]
        MAX_POS = max(max(AUX_POS), max(VERB_POS))

        if printPair:
            aux_words = " ".join([sentence[i-1]["text"] for i in aux_chunk])
            verb_words = " ".join([sentence[i-1]["text"] for i in verb_chunk])
            print("<{}, {}>".format(aux_words, verb_words))

        if len(aux_chunk) == 0: return result

        for pos, idx in enumerate(result):
            if idx in aux_chunk or pos > MAX_POS:
                # print(f"skipped {idx}")
                continue
            # print(f"this {idx} is being added")
            res.append(idx)
        res.extend([idx for idx in result if idx in aux_chunk])
        res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])

        return res

    def swap(self, sentence, root, printPair=False):
        """ DFS function for swapping AUX and VERB """
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
                    if sentence[c-1]['coarse_dep'] in Swapper.AUX_VERB_ARCS:
                        verb_idx, aux_idx = node - 1, c - 1
                        if aux_idx < verb_idx: # <aux, verb>
                            result = self.swap_pair(aux_idx, verb_idx, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return result


class NOUN_G_Swapper(Swapper):
    def __init__(self, order=1, space=True):
        super().__init__("NOUN_G", order, space)
    
    def swap_pair(self, noun_idx, g_idx, sentence, result, printPair=False):
        """ Helper function for swapping """
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []
        g_chunk = self.get_all_descendant(g_idx + 1, sentence)
        # print(g_chunk)
        g_chunk = [i for i in g_chunk if i <= g_idx + 1]
        # print(g_chunk)

        NOUN_POS = result.index(noun_idx + 1)
        G_POS = [result.index(i) for i in g_chunk]
        MAX_POS = max(G_POS)

        if printPair:
            noun_word = sentence[noun_idx]["text"]
            g_words = " ".join([sentence[i-1]["text"] for i in g_chunk])
            print("<{}, {}>".format(noun_word, g_words))

        for pos, idx in enumerate(result):
            if pos == NOUN_POS or pos > MAX_POS:
                continue
            res.append(idx)
        res.append(noun_idx+1)
        res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])

        return res
    
    def swap(self, sentence, root, printPair=False):
        """ DFS function for swapping NOUN and G """
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
                    if sentence[c-1]['coarse_dep'] in Swapper.NOUN_G_ARCS:
                        # print(f"The pair is {node} and {c}")
                        noun_idx, g_idx = node - 1, c - 1
                        if noun_idx < g_idx: # <Noun, Genitive>
                            result = self.swap_pair(noun_idx, g_idx, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return result


class COMP_S_Swapper(Swapper):
    def __init__(self, order=1, space=True):
        super().__init__("COMP_S", order, space)
    
    def swap_pair(self, comp_idx, s_idx, sentence, result, printPair=False):
        """ Helper function for swapping """
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []
        s_chunk = self.get_all_descendant(s_idx + 1, sentence)
        s_chunk.remove(comp_idx + 1)

        COMP_POS = result.index(comp_idx + 1)
        S_POS = [result.index(i) for i in s_chunk if i > COMP_POS]
        MAX_POS = max(S_POS)

        if printPair:
            comp_word = sentence[comp_idx]["text"]
            s_words = " ".join([sentence[i-1]["text"] for i in s_chunk])
            print("<{}, {}>".format(comp_word, s_words))

        for pos, idx in enumerate(result):
            if pos == COMP_POS or pos > MAX_POS:
                continue
            res.append(idx)
        res.append(comp_idx+1)
        res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])

        return res
    
    def swap(self, sentence, root, printPair=False):
        """ DFS function for swapping COMP and S """
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
                    if sentence[c-1]['coarse_dep'] in Swapper.COMP_S_ARCS:
                        # print(f"The pair is {node} and {c}")
                        s_idx, comp_idx = node - 1, c - 1
                        if comp_idx < s_idx: # <Complementizer, S>
                            result = self.swap_pair(comp_idx, s_idx, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return result