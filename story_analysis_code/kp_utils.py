import os, re, sys, json, string, gzip, csv
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from allennlp.predictors.predictor import Predictor
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

class GetPov:
    def __init__(self, datapath):
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",cuda_device=0)
        self.predictor._model = self.predictor._model.cuda()
        print(self.predictor.cuda_device)
        print("predictor loaded")
        with open(datapath, 'r') as f:
            lines = json.loads(f.read().strip())
            self.pov_map = lines
        print("get pov loaded")

    def getPronPreprocess(self, para):
        pron = []
        pron_p = []
        output = self.predictor.predict(
            document=para
        )
        if output['clusters'] != []:
            cluster_len = [len(i) for i in output['clusters']]
            for tup in output['clusters'][cluster_len.index(max(cluster_len))]: # finds cluster with max length (character that occurred most frequently)

                s = tup[0]
                t = tup[1]
                if s==t:
                    pron_p.append((s,s))
                else:
                    pron_p.append((s,t+1))
                pron.append(output['document'][s:t + 1])
        else:
            return None, None

        return pron, pron_p

    def get_pov(self, story):
        malePron = ["He","he","him","Him"]
        femalePron = ["She","she","Her","her"]
        gender = ''
        pron,pron_p = self.getPronPreprocess(story)
        if pron is None:
            pron = [pron]
        else:
            pron = [" ".join(i) for i in pron]
        if bool(set(pron) & set(malePron)) and not bool(set(pron) & set(femalePron)):
            gender = 'male'
        elif bool(set(pron) & set(femalePron)) and not bool(set(pron) & set(malePron)):
            gender = 'female'
        else:
            gender = 'unknown'
        return gender
    
    def get_pov_by_story(self, prompt_index, story_index, prompt, story):
        if len(story.split())>500:
            new_story = " ".join(story.split()[:500])
        else:
            new_story = story
        if prompt in self.pov_map: # strip
            vals = self.pov_map[prompt]
        elif ' ' + prompt in self.pov_map:
            vals = self.pov_map[" " + prompt]
        else:            
            return self.get_pov(new_story)
        
        stories, povs = vals[0], vals[-1]
        try:
            if story.strip() == stories[story_index].strip():
                human_pov = povs[story_index]
                return human_pov
            else:
                return self.get_pov(new_story)
        except:
            return self.get_pov(new_story)


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
    
    def get_words(self, trait):
        total = []
        with open(self.fast_lex_path, 'r') as infile: 
            for line in infile:
                contents = line.strip().split('\t') # intellect, smart, genius etc.
                for word in trait:
                    if word in contents[0]:
                        subwords = contents[1].split(',')
                        total.extend(subwords)
                    total.append(word.strip())
        # print("trait %s total %s:" % (trait, total))
        return list(set(total))

    def get_fast_lexicon(self):
        intellect = ["intellectual", "intuitive", "imaginative", "knowledgeable", "ambitious", "intelligent", "opinionated", "admirable", "eccentric", "crude", "likable", "empathetic", "superficial", "tolerant", "resourceful", "uneducated", "academically", "studious", "temperamental", "exceptional", "cynical", "outspoken", "destructive", "dependable", "amiable", "impulsive", "frivolous", "insightful", "overconfident", "charismatic", "prideful", "influential", "likeable", "unconventional", "educated", "flawed", "articulate", "pretentious", "perceptive", "vulgar", "easygoing", "listener", "skillful", "assertive", "philosophical", "rebellious", "selfless", "cunning", "deceptive", "artistic", "appalling", "overbearing", "temperament", "diligent", "charitable", "disposition", "quirky", "strategic", "compulsive", "benevolent", "pessimistic", "scientific", "flamboyant", "obsessive", "selective", "oriented", "humorous", "narcissistic", "reliable", "headstrong", "manipulative", "practical", "rewarding", "refined", "resilient", "desirable", "spiritual", "tendencies", "pompous", "judgmental", "respected", "inexperienced", "compassionate", "promiscuous", "argumentative", "conventional", "intellectually", "expressive", "impractical", "observant", "fickle", "hyperactive", "immoral", "straightforward", "vindictive"]
        appearence = ["beautiful","sexual"]
        appear = self.get_words(appearence)
        power = ["dominant","strong"]
        power = self.get_words(power)
        weak = ['submissive','weak','dependent','afraid']
        weak = self.get_words(weak)
        weak_vecs = [self.w2v_model.wv[i] for i in weak if i in self.w2v_model]
        power_vecs = [self.w2v_model.wv[i] for i in power if i in self.w2v_model]
        appear_vecs = [self.w2v_model.wv[i] for i in appear if i in self.w2v_model]
        intellect_vecs = [self.w2v_model.wv[i] for i in intellect if i in self.w2v_model]
        return weak_vecs, power_vecs, appear_vecs, intellect_vecs, appear, intellect

