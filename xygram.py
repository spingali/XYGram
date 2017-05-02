#!/usr/local/bin/python
import itertools, panphon, epitran
from scipy import spatial
import numpy as np
from sklearn.metrics import jaccard_similarity_score

class XYGram:
    features = [ 'syl', 'son', 'cont', 'nas', 'ant', 'cor', 'hi', 'lo', 'back' ]
    ft       = panphon.FeatureTable()

    def __init__(self, lang1, lang2, max_offset=3, max_features=3):
        self.epi          = (epitran.Epitran(lang1), epitran.Epitran(lang2))
        self.max_offset   = max_offset
        self.max_features = max_features

    def _allFeatureCombos(self, v):
        result = []
        for r in range(1, self.max_features + 1):
            result += list(itertools.combinations(v, r))
        return result

    # lang: 1 or 2 based on which of the two languages
    def generateXYGram(self, s, lang):
        epi = self.epi[lang - 1]
        ft_vector = XYGram.ft.word_array(XYGram.features, epi.transliterate(s))

        d = {}
        for i in range(len(s)):
            for j in range(i + 1, min(i + self.max_offset, len(s) + 1)):
                fv = ft_vector[i:j]
                tmp1 = [ [ k for k, x in enumerate(v) if x >= 0 ] for v in fv ]
                tmp2 = [ self._allFeatureCombos(v) for v in tmp1 ]
                keys = list(itertools.product(*tmp2))
                for k in keys:
                    d[k] = d.get(k, 0) + 1
        return d

    # Prereq: v1, v2 int lists of equal length
    def cosineSimilarity(self, v1, v2):
        if (len(v1) != len(v2)):
            raise ValueError
        return 1 - spatial.distance.cosine(v1, v2)

    # Prereq: v1, v2 int lists of equal length
    def jaccardSimilarity(self, v1, v2):
        if (len(v1) != len(v2)):
            raise ValueError
        return jaccard_similarity_score(v1, v2)

    def compareXYGram(self, xy1, xy2):
        # Vectorize dictionaries with same keys
        v1 = []
        v2 = []

        k1 = set(xy1.keys())
        k2 = set(xy2.keys())
        k  = k1.union(k2)
        for key in k:
            v1.append(xy1.get(key, 0))
            v2.append(xy2.get(key, 0))

        return self.jaccardSimilarity(v1, v2)

    def compareRaw(self, s1, s2):
        xy1 = self.generateXYGram(s1, 1)
        xy2 = self.generateXYGram(s2, 2)
        return self.compareXYGram(xy1, xy2)
