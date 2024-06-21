import os, re, sys, json, string, gzip, csv
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from gensim.models import KeyedVectors
from gensim import models

## function to read lexicon
def read_lexicon(lex_path):
    '''
    CSV file with header: [word, emotion]

    [could be implemented: remove the middle portion?]
    '''
    df = pd.read_csv(lex_path)
    # emotion = lex_path.split("/")[-1].replace(".csv", "")
    assert 'word' in df.columns
    assert 'score' in df.columns

    df.dropna(subset=['word', 'score'], inplace=True)
        

    w2v = {}
    for w, v in zip(df['word'], df['score']):
        w2v[w] = float(v)
    
    return w2v

## word2vec class
class WordEmbModel:
    def __init__(self, w2v_path) -> None:
        '''
        path to w2v file (.bin.gz)
        '''
        self.w2v_model = models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    
    def get(self, word):
        #lower case
        lw = word.lower()
        if lw in self.w2v_model:
            return self.w2v_model[lw]
        return None

