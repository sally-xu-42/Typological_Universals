from apply_counterfactual_grammar import initializeOrderTable
from itertools import product, combinations
import glob
import random
from collections import defaultdict
import json

if __name__ == "__main__":

    langs = ["en", "ja", "vi", "fr", "de"]
    partitions = ["train", "test", "valid"]
    seeds = [1, 2, 3]
    langmap = {
        "en": "English",
        "ja": "Japanese",
        "vi": "Vietnamese",
        "fr": "French",
        "de": "German",
    }

    dhWeights = defaultdict(dict)
    distanceWeights = defaultdict(dict)

    # construct random grammars using random seeds
    for lang, partition, seed in product(langs, partitions, seeds):
        filename = glob.glob(f"../data/wiki40b-txt-parsed/{lang}.{partition}.conllu")
        assert(len(filename) == 1)
        filename = filename[0]
        random.seed(seed)
        depsVocab = initializeOrderTable(filename, langmap[lang])
        itos_deps = sorted(depsVocab)
        for x in itos_deps:
            dhWeights[f"{lang}.{partition}.{seed}"][x] = random.random() - 0.5
            distanceWeights[f"{lang}.{partition}.{seed}"][x] = random.random() - 0.5

    # check that for a given language and seed, the random grammars are the same across partitions
    # this is crucial for a fair comparison so that the model is evaluated on test data from the same
    # grammar as the training data
    for lang, seed in product(langs, seeds):
        for part1, part2 in combinations(partitions, 2):

            d1 = dhWeights[f"{lang}.{part1}.{seed}"]
            d2 = dhWeights[f"{lang}.{part2}.{seed}"]
            if d1 != d2:
                print(json.dumps(d1, indent=4))
                print(json.dumps(d2, indent=4))
            d1 = distanceWeights[f"{lang}.{part1}.{seed}"]
            d2 = distanceWeights[f"{lang}.{part2}.{seed}"]
            if d1 != d2:
                print(json.dumps(d1, indent=4))
                print(json.dumps(d2, indent=4))