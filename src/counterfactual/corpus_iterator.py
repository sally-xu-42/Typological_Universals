# Iterating over a CoNNL-U dataset
# Original Author: Michael Hahn
# Adapted by: Tianyang Xu

# the length() and permute() functions were removed as these
# are not compatible with the iterator-based approach to file-reading

HEADER = [
    "index",
    "word",
    "lemma",
    "posUni",
    "posFine",
    "morph",
    "head",
    "dep",
    "_",
    "_",
]


def read_conllu_file(path):
    """Reads all data from a CoNNL-U format file and returns a list of 
    strings, each corresponding to a sentence in the dataset. Assumes 
    properly formed CoNNL-U format data.
    NOTE: this function is deprecated and we instead use an iterator
    over the data file to save memory.
    TODO: can I delete this?

    Args:
        path (str): path to file

    Returns:
        List[str]: list of strings, where each string is a CoNNL-U record
        for a sentence (one line per word, tab-separated fields)
    """
    with open(path) as f:
        data = f.read().strip()
    data = data.split("\n\n")
    return data


class CorpusIterator:
    def __init__(
        self, filename, language, partition="train", storeMorph=False,
    ):
        # store args
        self.storeMorph = storeMorph
        self.filename = filename
        self.partition = partition
        self.language = language

    def processSentence(self, sentence):
        """Process a given sentence string, returning a tuple of the 
        dependency info in Hahn's format and a boolean for whether the 
        sentence is the start of a new document. 
        The latter assumes the input string is the result of calling 
        .split("\n\n") on a .conllu file that contains # newdoc annotations
        and has blank lines between successive records

        Args:
            sentence (str): string representation of a conllu parse for a 
            single sentence. Comments will be ingored, except for "# newdoc"
            which tells the function to return True for the newdoc value in
            the return tuple.

        Returns:
            Tuple[Dict[str,str], bool]: a tuple of a dictionary representation
            of the dependency parse information and a boolean which is True if
            the given sentence is the start of a new document. This is useful
            for parsing a whole corpus and separating documents properly.
        """

        # split each parse into lines (one line per word)
        # split each line into fields (separated by tabs)
        # sentence = list of lists of fields
        sentence = list(map(lambda x: x.split("\t"), sentence.split("\n")))

        newdoc = False

        result = []
        for i in range(len(sentence)):
            if sentence[i][0].startswith("#"):
                if sentence[i][0].startswith("# newdoc"):
                    newdoc = True
                continue
            if "-" in sentence[i][0]:  # if it is NUM-NUM
                continue
            if "." in sentence[i][0]:
                continue

            # sentence = list of dicts, where each key is a field name (see HEADER)
            sentence[i] = dict([(y, sentence[i][x]) for x, y in enumerate(HEADER)])
            sentence[i]["head"] = int(sentence[i]["head"])
            sentence[i]["index"] = int(sentence[i]["index"])
            sentence[i]["word"] = sentence[i]["word"].lower()

            if self.storeMorph:
                sentence[i]["morph"] = sentence[i]["morph"].split("|")

            sentence[i]["dep"] = sentence[i]["dep"].lower()
            if self.language == "LDC2012T05" and sentence[i]["dep"] == "hed":
                sentence[i]["dep"] = "root"
            if self.language == "LDC2012T05" and sentence[i]["dep"] == "wp":
                sentence[i]["dep"] = "punct"

            result.append(sentence[i])
        return result, newdoc

    def getSentence(self, index):
        result = self.processSentence(self.data[index])
        return result

    def iterator(self, rejectShortSentences=False):
        """Yields one sentence at a time from the dataset.

        Args:
            rejectShortSentences (bool, optional): whether to reject short sentences. 
            Defaults to False.

        Yields:
            Tuple[dict, bool]: 
            1st member of Tuple: dict representation of a sentence
            2nd member of Tuple: True if the sentence is the beginning of a new document
        """
        with open(self.filename) as f_in:
            buffer = []
            for line in f_in:
                if line != "\n":
                    buffer.append(line)
                else:
                    sentence = "".join(buffer).strip()
                    buffer = []
                    yield (self.processSentence(sentence))