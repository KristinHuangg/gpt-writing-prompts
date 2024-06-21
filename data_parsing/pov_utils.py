import os, re, sys, json, string, gzip, csv
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from allennlp.predictors.predictor import Predictor
from gensim.models import KeyedVectors
from gensim import models

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
    
    def get_pov_catch_exception(self, story):
        malePron = ["He","he","him","Him", "his", "His", "Himself", "himself"]
        femalePron = ["She","she","Her","her", "herself", "Herself"]
        first_person_Pron = ["I", "Me", "My", "Mine", "Myself", "me", "my", "i", "mine", "myself"]
        second_person_Pron = ["You", "you", "Your", "your", "Yourself", "yourself"]
        gender = ''
        try:
            pron,pron_p = self.getPronPreprocess(story)
        except (RuntimeError, ValueError, AssertionError) as e:
            gender = 'unknown'
            return gender
        if pron is None:
            pron = [pron]
        else:
            pron = [" ".join(i) for i in pron]
        print("pron clustered: ", pron)
        first_person_count = 0
        for word in first_person_Pron:
            first_person_count += pron.count(word)
        second_person_count = 0
        for word in second_person_Pron:
            second_person_count += pron.count(word)
        if first_person_count > len(pron) // 2: # if 1st person pronoun is more than half of all pronouns
            gender = 'first_person'
        elif second_person_count > len(pron) // 2: 
            gender = 'second_person'
        elif bool(set(pron) & set(malePron)) and not bool(set(pron) & set(femalePron)):
            gender = 'male'
        elif bool(set(pron) & set(femalePron)) and not bool(set(pron) & set(malePron)):
            gender = 'female'
        else:
            gender = 'unknown'
        print("gender: ", gender)
        return gender

    def get_pov(self, story):
        malePron = ["He","he","him","Him", "his", "His"]
        femalePron = ["She","she","Her","her"]
        first_person_Pron = ["I", "Me", "My", "me", "my", "i"]
        second_person_Pron = ["You", "you", "Your", "your"]
        gender = ''
        pron,pron_p = self.getPronPreprocess(story)
        if pron is None:
            pron = [pron]
        else:
            pron = [" ".join(i) for i in pron]
        print("pron clustered: ", pron)
        first_person_count = 0
        for word in first_person_Pron:
            first_person_count += pron.count(word)
        second_person_count = 0
        for word in second_person_Pron:
            second_person_count += pron.count(word)
        if first_person_count > len(pron) // 2: # if 1st person pronoun is more than half of all pronouns
            gender = 'first_person'
        elif second_person_count > len(pron) // 2: 
            gender = 'second_person'
        elif bool(set(pron) & set(malePron)) and not bool(set(pron) & set(femalePron)):
            gender = 'male'
        elif bool(set(pron) & set(femalePron)) and not bool(set(pron) & set(malePron)):
            gender = 'female'
        else:
            gender = 'unknown'
        print("gender: ", gender)
        return gender
    
    def get_pov_by_prompt(self, prompt_index, prompt):
        return self.get_pov_catch_exception(prompt)
    
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

