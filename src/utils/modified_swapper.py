import string

class Swapper():
    """ Class for swapping a pair of specified grammatical elements """

    OBJ_ARCS = ["ccomp", "lifted_cop", "expl", "iobj", "obj", "obl", "xcomp"]
    VERB_POS = ["VERB", "AUX"]

    ADP_NP_ARCS = ["case"]
    ADP_POS = ["ADP"]
    NP_POS = ["NOUN", "PROPN", "NUM", "PRON"]

    COP_PRED_ARCS = ["lifted_cop"]

    AUX_VERB_ARCS = ["aux"]

    NOUN_G_ARCS = ["nmod"]

    NOUN_RELCL_ARCS = ["acl:relcl"]

    COMP_S_ARCS = ["mark"]

    COARSE_EXCEPTIONS = ["acl:relcl"]

    MONTHS = [
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

    def __init__(self, pair, order=1, space=True, upsample=False):

        self.pair = pair
        self.order = order
        self.space = space
        self.upsample = upsample

        self.SPECIAL_CC = True if pair in ["VO", "AUX_V", "ADP_NP", "COMP_S", "NOUN_G"] else False
        self.SPECIAL_COP = True if pair == ["VO", "AUX_V"] else False # don't need this line in this file actually
        self.SPECIAL_MARK = True if pair == "VO" else False # leave special case for mark in VO swapping
        self.SPECIAL_EXPL = True if pair == "VO" else False # leave special case for "there is" in VO swapping
        self.SPECIAL_ADVCL = True if pair == "VO" else False # leave special case for "there is" in VO swapping
        self.SPECIAL_CONJ = True if pair in ["AUX_V", "ADP_NP", "COMP_S", "NOUN_G"] else False # special case for conj

    def makeCoarse(self, x):
        if x in Swapper.COARSE_EXCEPTIONS: # exclude exceptions
            return x
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

    def sanity_dates(self, sentence):
        # Assuming 'sentence' is a list of dicts with 'text' and 'dep' keys, and 'head' being an index.
        for i, word in enumerate(sentence):
            if word['word'] in Swapper.MONTHS:
                # Check if the previous word is a number and if the head of current word (month) is the number.
                if i > 0 and sentence[i-1]['word'].isdigit() and word['head'] == i:
                    print("Stanza has made an error in dates")
                    # Assign the day's dependency to the month (making the month the head)
                    word["dep"] = sentence[i-1]["dep"]
                    sentence[i-1]['coarse_dep'] = 'nummod'  # or whatever the correct relation should be
                    word["head"] = sentence[i-1]['head']
                    sentence[i-1]['head'] = i+1 # Assuming 'head' is 1-indexed and pointing to the current month.
                    # Optionally, update the month's head if necessary, e.g., pointing to the year or 'ROOT'.
        return sentence
    
    def check_mark(self, verb_id, sentence):
        """ deal the case of mark moving with verb when mark is the head of an advcl
            input:
            verb_id: the verb id currently moving
            sentence: dictionary for sentence

            goal: don't record while as moving's child when going is the verb being moved.
            leave an exception for to
            while -> going -> south
            while -> south -> going
            return ids to be removed
        """
        res = set([])

        for c in sentence[verb_id-1]['children']:
            if sentence[c-1]["coarse_dep"] == "mark" and \
            sentence[c-1]["word"] != "to" and \
            (sentence[verb_id-1].get("advcl", False) or sentence[verb_id-1].get("split_mark", False)):
                res = self.get_all_descendant(c, sentence)

        return res
    
    def check_conjunction(self, child_id, parent_id, sentence):
        """ helper function for some swappers """
        if sentence[child_id-1]["coarse_dep"] == "conj" and \
            (not sentence[parent_id-1].get("has_conj", None)) and \
            (not sentence[child_id-1].get("conj", None)):
            print(f"{sentence[child_id-1]['word']} is not added to {sentence[parent_id-1]['word']}'s chunk") #
            return True
        else:
            return False
    
    def get_all_descendant(self, id, sentence):
        """ DFS function for getting all descendants with depth tracking """
        stack = [(id, 0)]
        res = []

        while stack:
            node, depth = stack.pop()
            if (node == 0) or (not sentence[node-1].get("children", None)):
                continue
            if self.SPECIAL_CONJ:
                for c in sentence[node-1]['children']:
                    if depth == 0 and self.check_conjunction(c, node, sentence):
                        continue
                    if depth == 0 and self.pair in ["AUX_V"] and sentence[c-1]["coarse_dep"] in ["advcl", "nsubj", "csubj"]:
                        continue
                    stack.append((c, depth+1)) # increment depth
                    res.append(c)
            else:
                for c in sentence[node-1]['children']:
                    stack.append((c, depth+1))
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
            if line["coarse_dep"] in ["nsubj", "csubj"]: # added csubj, not sure about examples with expl
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
            
            # CASE: <V,O> conjugation
            # change the head of the object if head is a later verb in sentence and has a prior conj verb
            # and the prior conj has no objs
            if self.SPECIAL_CC and \
            self.pair == "VO" and \
            line["coarse_dep"] in Swapper.OBJ_ARCS and \
            sentence[headIndex].get("has_conj", None):
                # forget the verbs are conjuncted if there is an object directly connected to the prior verb
                right_verb_idx = sentence[headIndex]["has_conj"]
                sentence[right_verb_idx]["conj"] = None
                sentence[headIndex]["has_conj"] = None

            if self.SPECIAL_CC and \
            self.pair == "VO" and \
            line["coarse_dep"] in Swapper.OBJ_ARCS and \
            sentence[headIndex].get("conj", None):
                # change the head verb if the prior conj verb has no obj
                line["head"] = sentence[headIndex]["conj"] + 1
                headIndex = sentence[headIndex]["conj"]
            
            # CASE: <AUX,V> conjugation
            if self.SPECIAL_CC and \
            self.pair == "AUX_V" and \
            line["coarse_dep"] in Swapper.AUX_VERB_ARCS and \
            sentence[headIndex].get("conj", None):
                # forget the prior verb has a conj if there is another aux directly connected to it
                left_aux_idx = sentence[headIndex]["conj"]
                sentence[left_aux_idx]["has_conj"] = None
                sentence[headIndex]["conj"] = None

            # CASE: <ADP,NP> conjugation
            if self.SPECIAL_CC and \
            self.pair == "ADP_NP" and \
            line["coarse_dep"] in Swapper.ADP_NP_ARCS and \
            sentence[headIndex].get("conj", None):
                # forget the prior noun phrase has a conj if there is another adposition directly connected to its conj
                # differenciate "x1 y1 and x2 y2" with "x y1 and y2" case
                left_noun_idx = sentence[headIndex]["conj"]
                sentence[left_noun_idx]["has_conj"] = None
                sentence[headIndex]["conj"] = None
            
            # CASE: <NOUN,G> conjugation
            # change the head of the genitive if head is a later noun in sentence and has a prior conj noun
            # and the prior conj has no genitives
            if self.SPECIAL_CC and \
            self.pair == "NOUN_G" and \
            line["coarse_dep"] == "case" and \
            sentence[headIndex]["dep"] in Swapper.NOUN_G_ARCS and \
            sentence[headIndex].get("has_conj", None):
                sentence[headIndex]["has_gen"] = True
                
            if self.SPECIAL_CC and \
            self.pair == "NOUN_G" and \
            line["coarse_dep"] == "case" and \
            sentence[headIndex]["dep"] in Swapper.NOUN_G_ARCS:
                grand_p = sentence[headIndex]["head"] - 1
                if sentence[grand_p].get("conj", None): # genitive of later noun
                    left_noun_idx = sentence[grand_p]["conj"]
                    if sentence[left_noun_idx].get("has_gen", None): # forget previous noun if it also has genitive
                        sentence[left_noun_idx]["has_conj"] = None
                        sentence[grand_p]["conj"] = None
                elif sentence[grand_p].get("has_conj", None) and sentence[grand_p].get("has_gen", None): # genitive of prev noun
                    right_noun_idx = sentence[grand_p]["has_conj"]
                    if headIndex >= right_noun_idx: # should be genitive of later noun, so change head to later noun
                        sentence[headIndex]["head"] = right_noun_idx + 1
                        sentence[grand_p]["has_conj"] = None
                        sentence[right_noun_idx]["conj"] = None
                
            # CASE: <COMP,S> conjugation
            # change the head of the clause head if clause head is head of a later clause in sentence and has a prior conj clause
            # and the later clause has no marks
            # TODO: don't need to care if left clause has a mark or not
            if self.SPECIAL_CC and \
            self.pair == "COMP_S" and \
            line["coarse_dep"] in Swapper.COMP_S_ARCS and \
            sentence[headIndex].get("has_conj", None):
                # put the first has_conj to None if there is mark attached
                right_clause_idx = sentence[headIndex]["has_conj"]
                sentence[headIndex]["has_conj"] = None

            if self.SPECIAL_CC and \
            self.pair == "COMP_S" and \
            line["coarse_dep"] in Swapper.COMP_S_ARCS and \
            sentence[headIndex].get("conj", None):
                # put the second conj to None if there is mark attached
                left_clause_idx = sentence[headIndex]["conj"]
                if sentence[left_clause_idx]["has_conj"] is None:
                    sentence[headIndex]["conj"] = None

            sentence[headIndex]["children"] = sentence[headIndex].get("children", []) + [line["index"]]
        return root, sentence

    def pipeline(self, sentence):
        if self.pair == "NOUN_G":
            sentence = self.sanity_dates(sentence)
        root, sentence = self.get_all_children(sentence)
        num_swaps, ordered = self.swap(sentence, root, printPair=True)
        if self.upsample and num_swaps == 0:
            # upsample process
            print("Skipped")
            return None
        else:
            output = self.idx_to_sent(ordered, sentence, self.space)
            return output
        
    def swap_pair(self):
        pass

    def swap(self):
        pass

   
class VOSwapper(Swapper):
    def __init__(self, order=1, space=True, upsample=False):
        super().__init__("VO", order, space, upsample)

    def swap_pair(self, verb_idx, obj_idx, subj, sentence, result, printPair=False):
        """ Helper function for swapping verb and object"""
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []
        verb_chunk = self.get_all_descendant(verb_idx + 1, sentence)
        obj_chunk = self.get_all_descendant(obj_idx + 1, sentence)
        subj_chunk = self.get_all_descendant(subj, sentence) # for human names like Arthur Ford
        verb_chunk = set([x for x in (verb_chunk - obj_chunk) if x > max(subj_chunk)])
        # check verb chunk to erase its mark head and desc
        verb_chunk = verb_chunk - self.check_mark(verb_idx + 1, sentence)
        
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
        num_swaps = 0
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
                            num_swaps += 1
                            result = self.swap_pair(verb_idx, obj_idx, subj, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return num_swaps, result


class ADP_NP_Swapper(Swapper):
    def __init__(self, order=1, space=True, upsample=False):
        super().__init__("ADP_NP", order, space, upsample)
    
    def swap_pair(self, adp_idx, np_idx, sentence, result, printPair=False):
        """ Helper function for swapping ADP and NP"""
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []

        if sentence[np_idx]["coarse_dep"] == "nummod":
            # This is a brute sanity check: Stanza error for wrong head of "case" in dates.
            # Example: 16 February 2016
            month_idx = sentence[np_idx]["head"] - 1
            if sentence[month_idx]["word"] in Swapper.MONTHS:
                print("Stanza has made an error in dates.")
                np_idx = month_idx

        adp_chunk = self.get_all_descendant(adp_idx + 1, sentence)
        np_chunk = self.get_all_descendant(np_idx + 1, sentence)

        if max(adp_chunk) >= max(np_chunk):
            np_chunk = set([x for x in (np_chunk - adp_chunk)])
        else:
            np_chunk = set([x for x in (np_chunk - adp_chunk) if x > max(adp_chunk)])

        ADP_POS = [result.index(i) for i in adp_chunk]
        NP_POS = [result.index(i) for i in np_chunk]

        if len(ADP_POS) == 0 or len(NP_POS) == 0: # sanity check, remove after created
            print(" ".join(x["word"] for x in sentence))
            return result
            printPair = True
        
        if printPair:
            adp_words = [sentence[i-1]["word"] for i in adp_chunk]
            np_words = [sentence[i-1]["word"] for i in np_chunk]
            print("<{}, {}>".format(adp_words, np_words))

        MAX_POS = max(max(ADP_POS), max(NP_POS))

        for pos, idx in enumerate(result):
            if idx in adp_chunk or pos > MAX_POS: continue
            res.append(idx)
        res.extend([idx for idx in result if idx in adp_chunk])
        res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])

        return res
    
    def swap(self, sentence, root, printPair=False):
        """ DFS function for swapping adposition and noun phrase """
        num_swaps = 0
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
                            if adp_idx < np_idx: # <ADP, NP>
                                num_swaps += 1
                                result = self.swap_pair(adp_idx, np_idx, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return num_swaps, result


class COP_PRED_Swapper(Swapper):
    def __init__(self, order=1, space=True, upsample=False):
        super().__init__("COP_PRED", order, space, upsample)

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
                continue
            res.append(idx)
        res.extend([idx for idx in result if idx in cop_chunk])
        res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])

        return res
    
    def swap(self, sentence, root, printPair=False):
        """ DFS function for swapping copula and predicate"""
        num_swaps = 0
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
                    if sentence[c-1]['coarse_dep'] in Swapper.COP_PRED_ARCS:
                        cop_idx, pred_idx = node - 1, c - 1
                        if cop_idx < pred_idx: # <cop, pred>
                            num_swaps += 1
                            result = self.swap_pair(cop_idx, pred_idx, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)
        
        # print(f"Total number of swaps is: {num_swap}")
        return num_swaps, result


class AUX_V_Swapper(Swapper):
    def __init__(self, order=1, space=True, upsample=False):
        super().__init__("AUX_V", order, space, upsample)
    
    def swap_pair(self, aux_idx, verb_idx, subj, sentence, result, printPair=False):
        """ Helper function for swapping """
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []
        aux_chunk = self.get_all_descendant(aux_idx + 1, sentence)
        verb_chunk = self.get_all_descendant(verb_idx + 1, sentence)
        subj_chunk = self.get_all_descendant(subj, sentence)

        verb_chunk = set([i for i in verb_chunk if (i > aux_idx + 1 and i > max(subj_chunk))])
        aux_chunk = set([x for x in (aux_chunk - verb_chunk) if (x > aux_idx and x > max(subj_chunk))])

        AUX_POS = [result.index(i) for i in aux_chunk]
        VERB_POS = [result.index(i) for i in verb_chunk]
        MAX_POS = max(max(AUX_POS), max(VERB_POS))

        if printPair:
            aux_words = " ".join([sentence[i-1]["word"] for i in aux_chunk])
            verb_words = " ".join([sentence[i-1]["word"] for i in verb_chunk])
            print("<{}, {}>".format(aux_words, verb_words))

        if len(aux_chunk) == 0: return result

        for pos, idx in enumerate(result):
            if idx in aux_chunk or pos > MAX_POS:
                continue
            res.append(idx)
        res.extend([idx for idx in result if idx in aux_chunk])
        res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])

        return res

    def swap(self, sentence, root, printPair=False):
        """ DFS function for swapping AUX and VERB """
        num_swaps = 0
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
                        subj = sentence[node-1].get('nsubj', 0)
                        subj_idx = subj - 1
                        if aux_idx < verb_idx and subj_idx < aux_idx: # <subj, aux, verb>
                            num_swaps += 1
                            result = self.swap_pair(aux_idx, verb_idx, subj, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return num_swaps, result


class NOUN_G_Swapper(Swapper):
    def __init__(self, order=1, space=True, upsample=False):
        super().__init__("NOUN_G", order, space, upsample)

    def get_noun_descendant(self, id, sentence):
        """ DFS function for getting all descendants of noun in <NOUN, G>, with depth tracking. """
        stack = [(id, 0)]  # Store node and depth as a tuple
        res = []

        while stack:
            node, depth = stack.pop()  # Unpack the node and its depth
            if node == 0 or not sentence[node-1].get("children", None):
                continue
            for c in sentence[node-1]['children']:
                # Check for conjunction and advcl only at the first layer (depth 0)
                if depth == 0 and sentence[c-1]["coarse_dep"] not in ["nummod", "compound", "appos", "flat"]:
                    continue
                stack.append((c, depth + 1))  # Increment depth for recursive calls
                res.append(c)

        return set([id] + res)
    
    def swap_pair(self, noun_idx, g_idx, sentence, result, printPair=False):
        """ Helper function for swapping """
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []
        noun_chunk = self.get_noun_descendant(noun_idx + 1, sentence)
        g_chunk = self.get_all_descendant(g_idx + 1, sentence)
        noun_chunk = [i for i in (noun_chunk - g_chunk) if i <= g_idx+1]

        NOUN_POS = [result.index(i) for i in noun_chunk]
        G_POS = [result.index(i) for i in g_chunk]
        MAX_POS = max(max(NOUN_POS), max(G_POS))

        if printPair:
            noun_words = " ".join([sentence[i-1]["word"] for i in noun_chunk])
            g_words = " ".join([sentence[i-1]["word"] for i in g_chunk])
            print("<{}, {}>".format(noun_words, g_words))

        for pos, idx in enumerate(result):
            if pos in NOUN_POS or pos > MAX_POS:
                continue
            res.append(idx)
        res.extend([idx for pos,idx in enumerate(result) if pos in NOUN_POS])
        res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])

        return res
    
    def swap(self, sentence, root, printPair=False):
        """ DFS function for swapping NOUN and G """
        num_swaps = 0
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
                        noun_idx, g_idx = node - 1, c - 1
                        marker = False
                        for i in sentence[g_idx].get("children", []): # indicator for <NOUN, G> in English is "of"
                            if sentence[i-1]['word'] == 'of':
                                marker = True

                        if noun_idx < g_idx and marker: # <Noun, Genitive>
                            num_swaps += 1
                            result = self.swap_pair(noun_idx, g_idx, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return num_swaps, result


class NOUN_RELCL_Swapper(Swapper):
    def __init__(self, order=1, space=True, upsample=False):
        super().__init__("NOUN_RELCL", order, space, upsample)

    def swap_pair(self, noun_idx, cl_idx, sentence, result, printPair=False):
        """ Helper function for swapping noun and relative clause"""
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []
        cl_chunk = self.get_all_descendant(cl_idx + 1, sentence)
        noun_chunk = self.get_all_descendant(noun_idx + 1, sentence)
        noun_chunk = [i for i in (noun_chunk - cl_chunk) if i >= noun_idx + 1]

        NOUN_POS = [result.index(i) for i in noun_chunk]
        CL_POS = [result.index(i) for i in cl_chunk]
        MAX_POS = max(max(NOUN_POS), max(CL_POS))

        if printPair:
            noun_words = " ".join([sentence[i-1]["word"] for i in noun_chunk])
            cl_words = " ".join([sentence[i-1]["word"] for i in cl_chunk])
            print("<{}, {}>".format(noun_words, cl_words))

        for pos, idx in enumerate(result):
            if pos in NOUN_POS or pos > MAX_POS:
                continue
            res.append(idx)
        res.extend([idx for pos,idx in enumerate(result) if pos in NOUN_POS])
        res.extend([idx for pos,idx in enumerate(result) if pos > MAX_POS])

        return res
    
    def swap(self, sentence, root, printPair=False):
        """ DFS function for swapping NOUN and RELCL """
        num_swaps = 0
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
                    if sentence[c-1]['coarse_dep'] in Swapper.NOUN_RELCL_ARCS:
                        noun_idx, cl_idx = node - 1, c - 1
                        if noun_idx < cl_idx: # <Noun, Relative Clause>
                            num_swaps += 1
                            result = self.swap_pair(noun_idx, cl_idx, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return num_swaps, result


class COMP_S_Swapper(Swapper):
    def __init__(self, order=1, space=True, upsample=False):
        super().__init__("COMP_S", order, space, upsample)
    
    def swap_pair(self, comp_idx, s_idx, sentence, result, printPair=False):
        """ Helper function for swapping """
        # result = [1,3,2,4,5], set(2,3), (4,5) => [1,4,5,3,2]
        res = []
        s_chunk = self.get_all_descendant(s_idx + 1, sentence)
        s_chunk.remove(comp_idx + 1)
        s_chunk = [i for i in s_chunk if i > (comp_idx + 1)]

        COMP_POS = result.index(comp_idx + 1)
        S_POS = [result.index(i) for i in s_chunk]
        MAX_POS = max(S_POS)

        if printPair:
            comp_word = sentence[comp_idx]["word"]
            s_words = " ".join([sentence[i-1]["word"] for i in s_chunk])
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
        num_swaps = 0
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
                        s_idx, comp_idx = node - 1, c - 1
                        if comp_idx < s_idx and sentence[comp_idx]["word"] != "to": # <Complementizer, S>
                            num_swaps += 1
                            result = self.swap_pair(comp_idx, s_idx, sentence, result, printPair)
                    if c not in visited:
                        stack.append(c)

        return num_swaps, result

