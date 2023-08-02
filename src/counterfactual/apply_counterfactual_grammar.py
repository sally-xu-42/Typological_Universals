# Applying a counterfactual grammar to a language dataset
# Original Author: Michael Hahn, Thomas Hikaru Clark (thclark@mit.edu)
# Adapted by: Tianyang Xu

# Usage example (running a grammar optimized for overall efficiency):
# python3 apply_counterfactual_grammar.py \
#   --language English \
#   --model 7580379440 \
#   --base_dir manual_output_funchead_two_coarse_lambda09_best_large \
#   --filename english_sample.conllu

from collections import defaultdict
import re
import sys
import random
import argparse
import os
from importlib_metadata import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn.functional
from torch.autograd import Variable
from corpus_iterator_funchead import CorpusIteratorFuncHead
import json
from iso_639 import lang_codes
from variant_name2id import name2id

# import spacy, neuralcoref
# from spacy.tokens import Doc

recursionlimit = sys.getrecursionlimit()
sys.setrecursionlimit(min(4000, 2 * recursionlimit))

NON_PARAM_ORDERS = ["REAL_REAL", "REVERSE", "SORT_FREQ", "SORT_FREQ_REV", "MIN_DL_PROJ"]


def makeCoarse(x):
    """Make coarse, i.e. chop off anything after the colon

    Args:
        x (str): input string

    Returns:
        str: input truncated from colon onwards
    """
    if ":" in x:
        return x[: x.index(":")]
    return x


def initializeOrderTable(filename, language):
    """Get the set of dependency relations used in this particular grammar

    Args:
        filename (str): path to file
        language (str): name of language, e.g. 'English'

    Returns:
        set: set of dependency relations, e.g. {'case', 'nmod'}
    """
    depsVocab = set()
    for sentence, newdoc in CorpusIteratorFuncHead(
        filename, language, validate=False
    ).iterator():
        for line in sentence:
            line["coarse_dep"] = makeCoarse(line["dep"])
            depsVocab.add(line["coarse_dep"])
    return depsVocab


def subtree_len(sentence, root_index):
    """Recursively calculate the size of the subtree in the given sentence
    that begins at root_index

    Args:
        sentence (list[dict[str,str]]): sentence object, list of field-value mappings
        root_index (int): 1-indexed index of root of this sentence

    Returns:
        _type_: _description_
    """
    length = 0
    children = [word for word in sentence if word["head"] == root_index]

    if len(children) == 0:
        return 1

    for child in children:
        length += subtree_len(sentence, child["index"])
    return length


def _linearize_mindl_rec(sentence, root_index, add_right):
    """Recursive helper function for linearizing a sentence to minimze DL

    Args:
        sentence (list[dict[str,str]]): sentence object, a list of field-value mappings
        root_index (int): the 1-indexed index of the root of the subtree under consideration
        add_right (bool): True if the current subtree is being added to the right of the root

    Returns:
        _type_: _description_
    """

    children = sorted(
        [word for word in sentence if word["head"] == root_index],
        key=lambda x: subtree_len(sentence, x["index"]),
    )

    subtrees = [[word for word in sentence if word["index"] == root_index]]

    # Very important and tricky:
    #   if there are an even number of children, start adding children on the opposite side from the link to parent
    #   if there are an odd number of children, start adding children on the same side as the link to parent
    # This ensures that the longest child (they are already sorted by length) gets added on the opposite side
    # as the link to parent (see Gildea and Temperley, 2007, p187)
    # Base case: one child. It should be added on the opposite side as the link to the parent, which
    # counterintuitively means that the recursive function should be called with the same value of add_right
    # as the call from the parent that led here.
    right = (not add_right) ^ (len(children) % 2 == 1)

    for child in children:
        if right:
            subtrees.append(_linearize_mindl_rec(sentence, child["index"], right))
        else:
            subtrees.insert(0, _linearize_mindl_rec(sentence, child["index"], right))
        right = not right

    x = list(itertools.chain.from_iterable(subtrees))
    return x


def linearize_mindl(sentence):
    """Linearize a dependency tree to minimize total DL under projectivity constraint.

    Args:
        sentence (list[dict[str,str]]): sentence object, a list of field-value mappings

    Raises:
        ValueError: if sentence does not have exactly one root

    Returns:
        list[dict[str,str]]: re-ordered sentence object (original indices preserved).
    """

    root = list(filter(lambda x: x["dep"] == "root", sentence))
    if len(root) != 1:
        raise ValueError(f"Expected exactly one root node, got {len(root)}.")
    root_index = root[0]["index"]
    new_sent = _linearize_mindl_rec(sentence, root_index, True)
    return new_sent


def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
    line = sentence[position - 1]
    allGradients = gradients_from_the_left_sum
    if "children_DH" in line:
        for child in line["children_DH"]:
            allGradients = recursivelyLinearize(sentence, child, result, allGradients)
    result.append(line)
    line["relevant_logprob_sum"] = allGradients
    if "children_HD" in line:
        for child in line["children_HD"]:
            allGradients = recursivelyLinearize(sentence, child, result, allGradients)
    return allGradients


def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax, distanceWeights):
    logits = [
        (x, distanceWeights[sentence[x - 1]["dependency_key"]])
        for x in remainingChildren
    ]
    logits = sorted(logits, key=lambda x: x[1], reverse=(not reverseSoftmax))
    childrenLinearized = list(map(lambda x: x[0], logits))
    return childrenLinearized


def orderSentence(sentence, model, dhWeights, distanceWeights, freqs=None, debug=False):
    """Linearize a sentence under a given grammar, parametrized by the given
    dependency-head directionality weights and distance weights

    Args:
        sentence (List[Dict[str->str]]): list of dicts mapping field names to values
        model (str): name of grammar, e.g. RANDOM or REAL_REAL or ID number
        dhWeights (Dict[str->float]): dictionary mapping UD deps to values
        distanceWeights (Dict[str->float]): dict mapping UD deps to values
        debug (bool, optional): whether to print verbose debug info. Defaults to False.

    Returns:
        List[Dict[str->str]]: _description_
    """
    root = None

    # for non-parametric orderings, some items will be eliminated (punctuation)
    if model in NON_PARAM_ORDERS:
        eliminated = []

    # iterate over lines in the parse (i.e. over words in the sentence)
    for line in sentence:

        # make the dependency relation label coarse (ignore stuff after colon)
        line["coarse_dep"] = makeCoarse(line["dep"])

        # identify the root
        if line["coarse_dep"] == "root":
            root = line["index"]
            continue

        # for non-parametric orderings, add punctuation to list of items to be eliminated
        # assumes that punctuation does not have non-punctuation dependents!
        if line["coarse_dep"].startswith("punct"):
            if model in NON_PARAM_ORDERS:
                eliminated.append(line)
            continue

        # this will be used later
        key = line["coarse_dep"]
        line["dependency_key"] = key

        # set the dependent-head directionality based on dhWeights
        direction = (
            "DH" if model in NON_PARAM_ORDERS or dhWeights.get(key) > 0 else "HD"
        )
        headIndex = line["head"] - 1
        sentence[headIndex]["children_" + direction] = sentence[headIndex].get(
            "children_" + direction, []
        ) + [line["index"]]

    # for non-parametric orderings, handle eliminations
    if model in NON_PARAM_ORDERS:
        while len(eliminated) > 0:
            line = eliminated[0]
            del eliminated[0]
            if "removed" in line:
                continue
            line["removed"] = True
            if "children_DH" in line:
                assert 0 not in line["children_DH"]
                eliminated = eliminated + [sentence[x - 1] for x in line["children_DH"]]
            if "children_HD" in line:
                assert 0 not in line["children_HD"]
                eliminated = eliminated + [sentence[x - 1] for x in line["children_HD"]]

        # a sentence is a list of dicts; filter out dicts that were removed
        linearized = list(filter(lambda x: "removed" not in x, sentence))

        # handle REVERSE (reverse of original order)
        if model == "REVERSE":
            linearized = linearized[::-1]

        # handle SORT_FREQ (highest to lowest word frequency)
        if model == "SORT_FREQ":
            linearized = sorted(
                linearized, key=lambda x: freqs.get(x["word"], 0.0), reverse=True
            )

        # handle SORT_FREQ_REV (lowest to highest word frequency)
        if model == "SORT_FREQ_REV":
            linearized = sorted(
                linearized, key=lambda x: freqs.get(x["word"], 0.0), reverse=False
            )

        # handle MIN_DL_PROJ (minimum dependency length with projectivity constraint)
        if model == "MIN_DL_PROJ":
            linearized = linearize_mindl(linearized)

    # for other orderings, order children using orderChildrenRelative()
    else:
        for line in sentence:
            if "children_DH" in line:
                childrenLinearized = orderChildrenRelative(
                    sentence, line["children_DH"][:], False, distanceWeights
                )
                line["children_DH"] = childrenLinearized
            if "children_HD" in line:
                childrenLinearized = orderChildrenRelative(
                    sentence, line["children_HD"][:], True, distanceWeights
                )
                line["children_HD"] = childrenLinearized

        # recursively linearize a sentence
        linearized = []
        recursivelyLinearize(
            sentence, root, linearized, Variable(torch.FloatTensor([0.0]))
        )

        # store new dependency links
        moved = [None] * len(sentence)
        for i, x in enumerate(linearized):
            moved[x["index"] - 1] = i
        for i, x in enumerate(linearized):
            if x["head"] == 0:  # root
                x["reordered_head"] = 0
            else:
                x["reordered_head"] = 1 + moved[x["head"] - 1]

    if debug:
        print("Original")
        print(" ".join(list(map(lambda x: x["word"], sentence))))
        print("Linearized")
        print(" ".join(list(map(lambda x: x["word"], linearized))))

    return linearized


def get_model_specs(args):
    """Retrieve the model specifications from the grammar descriptions file,
    or generate random grammar specifications if model=='RANDOM'

    Args:
        filename (str): path to dataset file
        model (str): name of model, e.g. 'REAL_REAL', 'RANDOM', or '7580379440'
        language (str): name of language, e.g. 'English'
        base_dir (str): path to directory containing grammar description file

    Returns:
        Tuple[dict, dict]: two dictionaries corresponding to 1) dependency-head weights
        and 2) distance weights
    """

    dhWeights, distanceWeights = {}, {}

    # Grammar can be one of the following:
    # a) RANDOM grammar - dhWeights and distanceWeights randomly initialized
    # b) REAL_REAL - factual order
    # c) an optimized grammar from a grammar file (denoted by a numeric ID)
    #    i) optimized for efficiency (joint predictability and parseability)
    #   ii) optimized for DLM (while still being a consistent grammar)
    # d) REVERSE - mirror image version of REAL_REAL
    # e) SORT_FREQ - tokens sorted by word frequency, high to low
    # f) SORT_FREQ_REV - tokens sorted by word frequency, low to high
    # g) MIN_DL_PROJ - tokens linearized to minimize total DL for each sentence

    # handle the grammar specification and populate the dhWeights and distanceWeights dicts
    if args.model in NON_PARAM_ORDERS:
        return dhWeights, distanceWeights
    if args.model.startswith("RANDOM"):  # a random ordering

        # read the grammar file
        grammar_file = os.path.join(args.base_dir, "auto-summary-lstm.tsv")
        df = pd.read_csv(grammar_file, sep="\t")

        # Get the list of unique deps in the file and assign random weights in [-0.5, 0.5]
        # It is crucial that the values and order of these are the same from run to run
        # so that the train/test/valid splits of RANDOM grammars will share params
        deps = sorted(set(df["CoarseDependency"].astype(str)))
        for x in deps:
            dhWeights[x] = random.random() - 0.5
            distanceWeights[x] = random.random() - 0.5
        sys.stderr.write("dhWeights\n" + json.dumps(dhWeights) + "\n")
        sys.stderr.write("distanceWeights\n" + json.dumps(distanceWeights) + "\n")
        return dhWeights, distanceWeights

    if args.model == "FREQ_OPT":
        grammar_file = os.path.join(args.base_dir_freqopt, f"{args.language}.tsv")
        df = pd.read_csv(grammar_file, sep="\t")
        dhWeights = dict(zip(df["Dependency"], map(float, df["DH_Weight"])))
        distanceWeights = dict(zip(df["Dependency"], map(float, df["DistanceWeight"])))
        return dhWeights, distanceWeights

    # if model is not in NON_PARAM_ORDERS or FREQ_OPT, it should be numeric ID
    if not args.model.isnumeric():
        raise ValueError(
            f"Model must be numeric or one of {NON_PARAM_ORDERS} but got {args.model}"
        )

    # load two sets of grammars - optimized, and approximations to real grammars
    grammars_optim = os.path.join(args.base_dir, "auto-summary-lstm.tsv")
    grammars_approx = os.path.join(args.base_dir_approx, "auto-summary-lstm.tsv")
    grammars_mindl = os.path.join(args.base_dir_mindl, "auto-summary-lstm.tsv")

    # combine into a single dataframe (account for difference in naming)
    grammars_optim = pd.read_csv(grammars_optim, sep="\t")
    grammars_approx = pd.read_csv(grammars_approx, sep="\t")
    grammars_mindl = pd.read_csv(grammars_mindl, sep="\t")
    grammars_approx.rename(
        columns={
            "DH_Mean_NoPunct": "DH_Weight",
            "Distance_Mean_NoPunct": "DistanceWeight",
            "Dependency": "CoarseDependency",
        },
        inplace=True,
    )
    grammars_mindl.rename(
        columns={"Dependency": "CoarseDependency",}, inplace=True,
    )
    df = pd.concat([grammars_optim, grammars_approx, grammars_mindl])

    # filter for parameters for this specific language and model ID
    df = df[(df["Language"] == args.language) & (df["FileName"] == int(args.model))]
    if len(df) == 0:
        l, m = args.language, args.model
        raise ValueError(f"Language or grammar not found: {l}, {m}")

    # get the weights
    dhWeights = dict(zip(df["CoarseDependency"], map(float, df["DH_Weight"])))
    distanceWeights = dict(
        zip(df["CoarseDependency"], map(float, df["DistanceWeight"]))
    )

    return dhWeights, distanceWeights


def get_dl(sentence):
    """Returns the summed dependency lengths for a sentence.

    Args:
        sentence (list[dict[str,any]]): sentence

    Returns:
        int: total dependency length of sentence
    """
    dl = 0
    # print("\n".join("\t".join(str(x) for x in word.values()) for word in sentence))
    for i, word in enumerate(sentence):
        if word["head"] == 0 or word["dep"] == "root":
            continue
        if "reordered_head" in word:
            dl += abs(word["reordered_head"] - (i + 1))
        else:
            dl += abs(word["head"] - (i + 1))
    return dl


def reorder_heads(sentence):
    """Adds a new field 'reordered_head' to a sentence object that maps the
    old head values to the 1-indexed indices of word positions in the sentence.

    Args:
        sentence (list[dict[str,any]]): sentence
    """
    mapping = dict((word["index"], i + 1) for i, word in enumerate(sentence))
    for word in sentence:
        if word["head"] != 0:
            word["reordered_head"] = mapping[word["head"]]
        else:
            word["reordered_head"] = 0


def coref_analysis(corpusIterator, nlp, dhWeights, distanceWeights, args):

    current_doc = []
    current_doc_positions = []
    coref_dists = []

    for i, (sentence, newdoc) in enumerate(corpusIterator):

        # upon starting a new document
        if newdoc and i > 0:

            # concatenate words in current document
            doc_str = " ".join(current_doc)
            doc = nlp(doc_str)
            clusters = doc._.coref_clusters

            # word position in spacy document -> word position in reordered document
            mapping = dict(
                zip(current_doc_positions, range(len(current_doc_positions)))
            )

            # for each coreference cluster
            for cluster in clusters:

                # add the start index (in transformed data) of each mention
                mention_starts = []
                for mention in cluster.mentions:
                    if mention.start in mapping:
                        mention_starts.append(mapping[mention.start])
                    else:
                        print("\t**", mention, mention.start)

                if len(mention_starts) < 2:
                    continue

                # sort by position and find the pairwise distances from
                # first mention to each subsequent mention
                mention_starts = sorted(mention_starts)
                for ms in mention_starts[1:]:
                    dist = abs(mention_starts[0] - ms)
                    if dist > 0 and dist <= 25:
                        coref_dists.append(dist)

            # reset for next document
            current_doc = []
            current_doc_positions = []

        # position in sentence + offset = position in document
        offset = len(current_doc)

        # this includes punctuation
        words = [x["word"] for x in sentence]
        current_doc.extend(words)

        # reordered sentence, this excludes punctuation
        ordered = list(
            orderSentence(sentence, args.model, dhWeights, distanceWeights, freqs)
        )
        current_doc_positions.extend([x["index"] - 1 + offset for x in ordered])

    # handle final document in dataset (not followed by #newdoc)
    if len(current_doc) > 0:
        doc_str = " ".join(current_doc)
        doc = nlp(doc_str)
        clusters = doc._.coref_clusters

        mapping = dict(zip(current_doc_positions, range(len(current_doc_positions))))

        for cluster in clusters:

            mention_starts = []
            for mention in cluster.mentions:
                if mention.start in mapping:
                    mention_starts.append(mapping[mention.start])
                else:
                    print("\t**", mention, mention.start)

            if len(mention_starts) < 2:
                continue

            mention_starts = sorted(mention_starts)
            for ms in mention_starts[1:]:
                dist = abs(mention_starts[0] - ms)
                if dist > 0 and dist <= 25:
                    coref_dists.append(dist)

    # print(coref_dists)
    print("mean coref dist:", np.mean(coref_dists), np.std(coref_dists))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", help="name of language, e.g. English", default="English"
    )
    parser.add_argument(
        "--model",
        help="type of model, e.g. RANDOM, REAL_REAL, or numerical ID of grammar",
        default="REAL_REAL",
    )
    parser.add_argument(
        "--base_dir",
        help="base directory of grammar file",
        default="grammars/manual_output_funchead_two_coarse_lambda09_best_large",
    )
    parser.add_argument(
        "--base_dir_approx",
        help="base directory of grammar file for approximations to real grammars",
        default="grammars/manual_output_funchead_ground_coarse_final",
    )
    parser.add_argument(
        "--base_dir_mindl",
        help="base directory of file for grammars optimized for DLM",
        default="grammars/manual_output_funchead_coarse_depl_balanced",
    )
    parser.add_argument(
        "--base_dir_freqopt",
        help="base directory for grammars optimized for high-frequency first",
        default="../grammars/freq_opt",
    )
    parser.add_argument(
        "--filename",
        help="filename of CONLLU data",
        default="../data/wiki40b-txt-parsed/en.tiny.conllu",
    )
    parser.add_argument(
        "--seed", help="random seed for making RANDOM grammars", type=int, default=1
    )
    parser.add_argument(
        "--output_dl_only",
        action="store_true",
        help="if set, will only output avg dependency length for a dataset/grammar",
    )
    parser.add_argument("--freq_opt", action="store_true")
    parser.add_argument("--coref_analysis", action="store_true")
    parser.add_argument(
        "--freq_dir",
        default="freqs",
        help="directory containing word frequency data for a language",
    )
    parser.add_argument("--debug_dl", action="store_true")
    args = parser.parse_args()

    # check args
    if not (args.language in lang_codes.keys() or args.language in lang_codes.values()):
        raise ValueError(f"Specified language is invalid: {args.language}")
    if args.language in lang_codes.keys():
        args.language = lang_codes[args.language]
    lang_codes_inv = {v: k for k, v in lang_codes.items()}

    if args.freq_opt:
        assert args.model == "REAL_REAL"

    # convert convenience name to numeric ID, if applicable
    lang_code = lang_codes_inv[args.language]
    if lang_code not in name2id.keys():
        raise ValueError(f"Specified language is invalid: {args.language}")
    if args.model in name2id[lang_code].keys():
        args.model = name2id[lang_code][args.model]

    # load frequencies
    freqs = {}
    if args.model in ["SORT_FREQ", "SORT_FREQ_REV"] or args.freq_opt:
        freq_path = os.path.join(args.freq_dir, f"{lang_codes_inv[args.language]}.csv")
        freqs = pd.read_csv(freq_path)
        freqs = freqs.groupby("word").sum().reset_index()
        freqs = dict(zip(freqs.word, freqs.freq))

    # handle RANDOM-XX
    if args.model.startswith("RANDOM"):
        assert re.match("^RANDOM_\d+$", args.model), f"Invalid model: {args.model}"
        args.seed = int(args.model.split("_")[-1])

    # get model specs from file if applicable
    random.seed(args.seed)
    dhWeights, distanceWeights = get_model_specs(args)

    # load and iterate over a corpus
    corpus = CorpusIteratorFuncHead(
        args.filename, args.language, "train", validate=False
    )
    corpusIterator = corpus.iterator()

    # For debugging dependency length minimization
    if args.debug_dl:
        sentence, newdoc = next(corpusIterator)
        print("\n".join("\t".join(str(x) for x in word.values()) for word in sentence))
        reorder_heads(sentence)
        print(get_dl(sentence))
        print()

        sentence = linearize_mindl(sentence)
        print("\n".join("\t".join(str(x) for x in word.values()) for word in sentence))
        reorder_heads(sentence)
        print(get_dl(sentence))
        quit()

    # generate a grammar that is optimized for putting high-frequency words early in sent
    if args.freq_opt:

        rel_weights = defaultdict(list)

        for i, (sentence, newdoc) in enumerate(corpusIterator):
            ordered = list(
                orderSentence(sentence, args.model, dhWeights, distanceWeights, freqs)
            )
            idx2word = dict(
                zip([w["index"] for w in ordered], [w["word"] for w in ordered])
            )

            for j, row in enumerate(ordered):
                if row["head"] == 0:
                    continue
                head_word = idx2word[row["head"]]
                head_freq = freqs.get(head_word, 0.0)
                child_freq = freqs.get(row["word"], 0.0)
                dep_name = makeCoarse(row["dep"])
                rel_weights[dep_name].append(1 if head_freq < child_freq else -1)

        rel_weights_final = {k: np.mean(v) for k, v in rel_weights.items()}

        df = pd.DataFrame(
            {
                "Language": args.language,
                "Dependency": rel_weights_final.keys(),
                "DH_Weight": rel_weights_final.values(),
                "DistanceWeight": rel_weights_final.values(),
            }
        )
        df.to_csv(f"{args.base_dir_freqopt}/{args.language}.tsv", index=False, sep="\t")
        quit()

    # For coreference analysis - currently disabled
    if args.coref_analysis:
        # nlp = spacy.load("en_core_web_sm")

        # def custom_tokenizer(text):
        #     return Doc(nlp.vocab, text.split())

        # nlp.tokenizer = custom_tokenizer
        # neuralcoref.add_to_pipe(nlp, max_dist=50, max_dist_match=50, greedyness=0.5)
        # coref_analysis(corpusIterator, nlp, dhWeights, distanceWeights, args)
        quit()

    # For outputting dependency length of datasets under different grammars
    if args.output_dl_only:
        dep_lens, sent_lens = [], []
        for i, (sentence, newdoc) in enumerate(corpusIterator):
            ordered = list(
                orderSentence(sentence, args.model, dhWeights, distanceWeights, freqs)
            )
            if args.model in NON_PARAM_ORDERS:
                reorder_heads(ordered)
            dep_lens.append(get_dl(ordered))
            sent_lens.append(len(ordered))
        sys.stdout.write(f"Language: {args.language}, Model: {args.model}\n")
        sys.stdout.write(f"Avg Sentence Length: {np.mean(sent_lens)}\n")
        sys.stdout.write(f"Avg Dependency Length: {np.mean(dep_lens)}\n")
        quit()

    # DEFAULT BEHAVIOR
    # iterate over all sentences in corpus
    for i, (sentence, newdoc) in enumerate(corpusIterator):
        ordered = list(
            orderSentence(sentence, args.model, dhWeights, distanceWeights, freqs)
        )
        output = " ".join(list(map(lambda x: x["word"], ordered)))

        # Add a new line if the just-processed sentence starts a new document
        if newdoc and i != 0:
            sys.stdout.write("\n")
        sys.stdout.write(output)
        sys.stdout.write(" . ")  # add a period after every sentence