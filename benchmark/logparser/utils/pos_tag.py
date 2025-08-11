from nltk.tag import *
from nltk.tag.perceptron import PerceptronTagger, AveragedPerceptron
from collections import defaultdict

PICKLE = "averaged_perceptron_tagger.pickle"


class PerceptronScoreTagger(PerceptronTagger):
    def __init__(self, load=True):
        self.model = AveragedScorePerceptron()
        self.tagdict = {}
        self.classes = set()
        if load:
            AP_MODEL_LOC = "file:" + str(
                find("taggers/averaged_perceptron_tagger/" + PICKLE)
            )
            self.load(AP_MODEL_LOC)

    def tag(self, tokens, return_conf=False, use_tagdict=True):
        """
        Tag tokenized sentences.
        :params tokens: list of word
        :type tokens: list(str)
        """
        prev, prev2 = self.START
        output = []

        context = self.START + [self.normalize(w) for w in tokens] + self.END
        for i, word in enumerate(tokens):
            tag, conf = (
                (self.tagdict.get(word), 1.0) if use_tagdict == True else (None, None)
            )
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                tag, conf = self.model.predict(features, return_conf)
            else:
                print(tag)
            output.append((word, tag, conf) if return_conf == True else (word, tag))

            prev2 = prev
            prev = tag

        return output

    def score_tag(self, tokens, return_conf=False, use_tagdict=True):
        prev, prev2 = self.START
        output = []

        context = self.START + [self.normalize(w) for w in tokens] + self.END
        for i, word in enumerate(tokens):
            features = self._get_features(i, word, context, prev, prev2)
            result = self.model.score_predict(features, return_conf)
            output.append(result)

        return output


class AveragedScorePerceptron(AveragedPerceptron):
    def __init__(self, weights=None):
        AveragedPerceptron.__init__(self, weights)

    def predict(self, features, return_conf=False):
        """Dot-product the features and current weights and return the best label."""
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight

        vmax = 0
        nmax = 0
        ntag = ""
        vtag = ""
        for i in dict(scores):
            if ("V" in i) and (scores[i] > vmax):
                vmax = scores[i]
                vtag = i
            if ("N" in i) and (scores[i] > nmax):
                nmax = scores[i]
                ntag = i
        print("V: {} {} N: {} {}".format(vmax, vtag, nmax, ntag))
        # Do a secondary alphabetic sort, for stability
        best_label = max(self.classes, key=lambda label: (scores[label], label))
        # compute the confidence
        conf = max(self._softmax(scores)) if return_conf == True else None

        return best_label, conf

    def score_predict(self, features, return_conf=False):
        """Dot-product the features and current weights and return the best label."""
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight

        vmax = 0
        nmax = 0
        # ntag = ""
        # vtag = ""
        # result = {
        #     "V": {},
        #     "N": {}
        # }
        for i in dict(scores):
            if ("V" in i) and (scores[i] > vmax):
                vmax = scores[i]
            if ("N" in i) and (scores[i] > nmax):
                nmax = scores[i]

        return vmax+nmax


def _pos_tag(tokens, tagset=None, tagger=None, lang=None):
    # Currently only supports English and Russian.
    if lang not in ["eng", "rus"]:
        raise NotImplementedError(
            "Currently, NLTK pos_tag only supports English and Russian "
            "(i.e. lang='eng' or lang='rus')"
        )
    # Throws Error if tokens is of string type
    elif isinstance(tokens, str):
        raise TypeError('tokens: expected a list of strings, got a string')

    else:
        tagged_tokens = tagger.tag(tokens)
        if tagset:  # Maps to the specified tagset.
            if lang == "eng":
                tagged_tokens = [
                    (token, map_tag("en-ptb", tagset, tag))
                    for (token, tag) in tagged_tokens
                ]
            elif lang == "rus":
                # Note that the new Russion pos tags from the model contains suffixes,
                # see https://github.com/nltk/nltk/issues/2151#issuecomment-430709018
                tagged_tokens = [
                    (token, map_tag("ru-rnc-new", tagset, tag.partition("=")[0]))
                    for (token, tag) in tagged_tokens
                ]
        return tagged_tokens


def _pos_score_tag(tokens, tagset=None, tagger=None, lang=None):
    result = tagger.score_tag(tokens)
    return result
