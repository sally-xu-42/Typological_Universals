# Applying a counterfactual grammar to a language dataset
# Original Author: Michael Hahn
# Adapted by: Tianyang Xu

# Usage example (running a grammar optimized for overall efficiency):
# python3 apply_counterfactual_grammar.py \
#   --language English \
#   --model RANDOM_3 \
#   --base_dir manual_output_funchead_two_coarse_lambda09_best_large \
#   --filename english_sample.conllu
#   --output en_approx.txt

import re
import sys
import random
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional
from torch.autograd import Variable
from corpus_iterator_funchead import CorpusIteratorFuncHead
import json
from iso_639 import lang_codes
from variant_name2id import name2id

recursionlimit = sys.getrecursionlimit()
sys.setrecursionlimit(min(4000, 2 * recursionlimit))

NON_PARAM_ORDERS = ["REAL_REAL"]
NON_SPACE_LANG = ["zh-cn", "ja", "ko"]


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

# TODO:read this. Recursive traversal?
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


def orderSentence(sentence, model, dhWeights, distanceWeights, debug=False):
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
    # keep track of the root
    root = None

    # for non-parametric orderings, some items will be eliminated (punctuation)
    if model in NON_PARAM_ORDERS:
        # list of word's dicts
        eliminated = []

    # iterate over lines in the parse (i.e. over words in the sentence)
    for line in sentence:

        # make the dependency relation label coarse (ignore stuff after colon)
        line["coarse_dep"] = makeCoarse(line["dep"])

        # identify the root, and skip to next word
        if line["coarse_dep"] == "root":
            root = line["index"]
            continue

        # for non-parametric orderings, add punctuation to list of items to be eliminated
        # assumes that punctuation does not have non-punctuation dependents!
        if line["coarse_dep"].startswith("punct"):
            if model in NON_PARAM_ORDERS:
                eliminated.append(line)
            continue

        # this will be used later, a coarse deprel, not being root or punct
        key = line["coarse_dep"]
        line["dependency_key"] = key

        # set the dependent-head directionality based on dhWeights
        # add line to its head's children list
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

        # recursively linearize a sentence, after every word's children list is permuted
        linearized = []
        # TODO: gradient is the variable...but why? what will happen if I change this tensor to 100?
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
    # d) REV grammar - reversing an existing pair of <X, Y>
    # only b) and d) are non-parametric

    # handle the grammar specification and populate the dhWeights and distanceWeights dicts
    if args.model in NON_PARAM_ORDERS:
        return dhWeights, distanceWeights
    if args.model.startswith("RANDOM"):  # a random ordering

        # read the efficient grammar file only to get deprels
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

    # if model is not in NON_PARAM_ORDERS, it should be numeric ID
    if not args.model.isnumeric():
        raise ValueError(
            f"Model must be numeric or one of {NON_PARAM_ORDERS} but got {args.model}"
        )

    # load three sets of grammars - optimized, approximations to real grammars, and min-DL grammar
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", help="name of language, e.g. English", default="English"
    )
    parser.add_argument(
        "--model",
        help="type of model, e.g. RANDOM, REAL_REAL, APPROX, or numerical ID of grammar",
        default="REAL_REAL",
    )
    parser.add_argument(
        "--base_dir",
        help="base directory of grammar file",
        default="../../grammars/manual_output_funchead_two_coarse_lambda09_best_large",
    )
    parser.add_argument(
        "--base_dir_approx",
        help="base directory of grammar file for approximations to real grammars",
        default="../../grammars/manual_output_funchead_ground_coarse_final",
    )
    parser.add_argument(
        "--base_dir_mindl",
        help="base directory of file for grammars optimized for DLM",
        default="../../grammars/manual_output_funchead_coarse_depl_balanced",
    )
    parser.add_argument(
        "--filename",
        help="filename of CONLLU data",
        default="../../parse/en_tiny.conllu",
    )
    parser.add_argument(
        "--output",
        help="output file of plain reordered text according to counterfactual grammar applied",
        default="../../en_random.txt"
    )
    parser.add_argument(
        "--seed", help="random seed for making RANDOM grammars", type=int, default=1
    )
    parser.add_argument(
        "--output_dl_only",
        action="store_true",
        help="if set, will only output avg dependency length for a dataset/grammar",
    )
    args = parser.parse_args()

    # check args
    if not (args.language in lang_codes.keys() or args.language in lang_codes.values()):
        raise ValueError(f"Specified language is invalid: {args.language}")
    if args.language in lang_codes.keys():
        args.language = lang_codes[args.language]
    lang_codes_inv = {v: k for k, v in lang_codes.items()}

    # convert convenience name like "APPROX" to numeric model ID, if applicable
    lang_code = lang_codes_inv[args.language]
    if lang_code not in name2id.keys():
        raise ValueError(f"Specified language doesn't exists in our models: {args.language}")
    if args.model in name2id[lang_code].keys():
        args.model = name2id[lang_code][args.model]

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

    # For outputting dependency length of datasets under different grammars
    if args.output_dl_only:
        dep_lens, sent_lens = [], []
        for i, (sentence, newdoc) in enumerate(corpusIterator):
            ordered = list(
                orderSentence(sentence, args.model, dhWeights, distanceWeights)
            )
            if args.model in NON_PARAM_ORDERS:
                reorder_heads(ordered)
            dep_lens.append(get_dl(ordered))
            sent_lens.append(len(ordered))
        with open(args.output, "w") as file:
            file.write(f"Language: {args.language}, Model: {args.model}\n")
            file.write(f"Avg Sentence Length: {np.mean(sent_lens)}\n")
            file.write(f"Avg Dependency Length: {np.mean(dep_lens)}\n")
        quit()

    # DEFAULT BEHAVIOR
    # iterate over all sentences in corpus
    with open(args.output, "w") as file:
        for i, (sentence, newdoc) in enumerate(corpusIterator):
            ordered = list(
                orderSentence(sentence, args.model, dhWeights, distanceWeights)
            )
            # don't connect by space in Chinese, Japanese and Korean
            if lang_code not in NON_SPACE_LANG:
                output = " ".join(list(map(lambda x: x["word"], ordered)))
            else:
                output = "".join(list(map(lambda x: x["word"], ordered)))

            # Add a new line if the just-processed sentence starts a new document
            # TODO: how about new paragraph?
            if newdoc and i != 0:
                file.write("\n")
            file.write(output)
            file.write(" . ")  # add a period after every sentence