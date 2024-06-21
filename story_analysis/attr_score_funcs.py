import os, re, sys, json, string, csv, gzip
import nltk
from scipy import spatial, stats
import pandas as pd
import spacy
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
try:
    from comet2.comet_model import PretrainedCometModel
except:
    pass

class TextToAttrs:

    def __init__(self, prot=['protagonistA', 'ProtagonistA'], spacy_model='en_core_web_lg', attr_method='full'):
        '''
        method: 'all': just return all tokens
                'sub': only amods descended from prot, and {VBX, JJ} heads of prot.  (protagonist as subject)
            (probabaly not the best names)
                'comet': return inferred intelligence, appearance, power attributes using comet commonsense 
        '''
        self.attr_method = attr_method
        self.model = spacy.load(spacy_model)
        self.prot = prot
        self.comet_model = None # we are using a pre-extracted list of comet attributes. Load only if necessary
        # ALTERNATE
        # if attr_method == 'comet':
        #     self.comet_model = PretrainedCometModel(device=0, model_name_or_path='~/comet-commonsense/comet-data/models/atomic_pretrained_model_openai-gpt/')


    def get_sub_attributes(self, doc, prot):
        attrs = []
        for sent in doc.sents:
            for tok in sent:
                if (tok.dep_ == 'nsubj') and (tok.text in prot):
                    head = tok.head
                    if head:
                        if head.tag_.startswith('VB') or head.tag_ == 'JJ':
                            attrs.append(head.text)
                elif tok.dep_ == 'amod':
                    for an in tok.ancestors:
                        if an.text in prot:
                            attrs.append(tok.text)
                            break
        return attrs


    def get_comet_inferences(self, doc, prot):

        prot_atts = []

        for sent in doc.sents:
            use = False
            sent_attrs = []
            for token in sent:
                if token.dep_ == 'ROOT':
                    for child in token.children:
                        if child.dep_ in ['nsubj', 'nsubjpass'] and child.text in prot:
                            use = True
                            break
            if use:
                try:
                    sent_attrs = self.comet_model.predict(sent.text, "xAttr", num_beams=5, length = 1024)
                except:
                    pass
            prot_atts.extend(sent_attrs)
        return prot_atts
    
    def comet_prep(self, attr):
        attr = ''.join(c for c in attr if c not in string.punctuation)
        attr = attr.lower()
        attr = attr.split()
        return attr
    

    # helper
    def get_attrs(self, text, prot=None, attr_method=None):
        if not attr_method:
            attr_method = self.attr_method

        if attr_method not in ['full', 'sub', 'all', 'comet']:
            attr_method = self.attr_method
            print("Invalid method, defaulting to {}".format(attr_method))
        
        if not prot:
            prot = self.prot
        

        attrs = []

        if attr_method == 'comet':
            # we are passing the list of attributes already
            toks = [self.comet_prep(x) for x in text]
            attrs = [x for sublist in toks for x in sublist]
            return attrs

            # ALTERNATE: run inference (takes long for the full dataset)
            # doc = self.model(text)
            # attrs = self.get_comet_inferences(doc, prot)
            

        doc = self.model(text)
        
        if attr_method=='sub':
            attrs = self.get_sub_attributes(doc, prot)

        else: # attr_method == all
            attrs = [x.text for x in doc]
        
        return attrs

    
class AttrsToScore:

    def __init__(self, lexicon=None, emb_model=None, method='avg'):
        '''
        lexicon (dict): word to score for a particular dimension 
        emb_model (dict): word to emb vector

        method: 'avg' -> average of lexicon[word] for word in text / 
                'sim': mean of [average of cosine(emb(word), emb(lex_word)) for each lex_word in lexicon] for each word in text / 
                'axis': mean of [cosine(emb(word), axis] for word in text, where axis is constructed as a vector direction representing the lexicon dimension 

        'axis' can only be used when lexicon contains terms that represent both (extreme) ends of the dimension, or positive and negative terms representing the dimension.
        '''

        self.lexicon = lexicon
        self.emb_model = emb_model
        self.method = method
        
    ####################### lexical average ########################

    def get_lex_avg(self, attrs):
        '''
        average across all the words according to the lexicon scores
        if lexicon only contains a single score (1 if associated), then compute average density of lexicon words
        '''
        assert self.lexicon is not None
        scores = []

        lex_vals = set(self.lexicon.values())

        if len(lex_vals) == 1:
            scores = [self.lexicon.get(x.lower(), 0) for x in attrs]
        else:
            scores = [self.lexicon.get(x.lower()) for x in attrs]
    
        filtered_scores = list(filter(lambda item: item is not None, scores))
        
        if len(filtered_scores) == 0:
            return 0
        
        return sum(filtered_scores)/len(filtered_scores)

    ####################### vector axis ########################
    # optional: select synonym-y pairs from extremes
    def get_sim_matrix(self, low_embs, high_embs):

        low_high_sim = cosine_similarity(low_embs, high_embs)
        return low_high_sim
        
    def get_contrast_pairs(self, div_terms, sim_matrix):
        div_inds = [set(), set()]
        reverse = False
        if len(div_terms[0]) > len(div_terms[1]):
            reverse = True
            div_terms = [div_terms[1], div_terms[0]]
            sim_matrix = np.transpose(sim_matrix)
        sim_pairs = []
        for ind1, row1 in enumerate(sim_matrix):

            sim_inds = np.argsort(row1)[::-1]
            i = 0
            while sim_inds[i] in div_inds[1]:
                i += 1
            sim_pairs.append([div_terms[0][ind1], div_terms[1][sim_inds[i]], row1[sim_inds[i]]])
            div_inds[0].add(ind1)
            div_inds[1].add(sim_inds[i])

        if reverse:
            sim_pairs = [[x[1], x[0], x[-1]] for x in sim_pairs]
        sim_pairs = sorted(sim_pairs, key=lambda x: x[-1])[::-1]
        return sim_pairs

    # construct axis
    def construct_axis(self, use_contrast=True):
        '''
        construct the vector axis

        create a set of low and high terms
        '''
        assert self.emb_model is not None

        scores = list(self.lexicon.values())
        # only consider terms on the extremes -- beyond mean +- 2*std_dev
        # mean, std = np.mean(scores), np.std(scores)
        # lb, ub = (mean - 2*std), (mean + 2*std)
        # use percentiles to allow for categorical dists
        lb = np.percentile(scores, 25)
        ub  = np.percentile(scores, 75)

        low_t = [k for k,v in self.lexicon.items() if v <= lb]
        low_t = [k for k in low_t if k in self.emb_model.w2v_model]
        high_t = [k for k,v in self.lexicon.items() if v>=ub]
        high_t = [k for k in high_t if k in self.emb_model.w2v_model]

        
        low_e = [self.emb_model.get(w) for w in low_t]
        # low_e = [x for x in low_e if x is not None]
        
        high_e = [self.emb_model.get(w) for w in high_t]
        # high_e = [x for x in high_e if x is not None]

        if use_contrast:
            sim_matrix = self.get_sim_matrix(low_e, high_e)

            contrast_pairs = self.get_contrast_pairs([low_t, high_t], sim_matrix)
            print("Found {} contrast pairs".format(len(contrast_pairs)))
            print(contrast_pairs[:5])
            print()

            low_pole = [x[0] for x in contrast_pairs]
            high_pole = [x[1] for x in contrast_pairs]
            low_e = [self.emb_model.get(w) for w in low_pole]
            high_e = [self.emb_model.get(w) for w in high_pole]

        low_mean = np.mean(np.array(low_e), axis=0)
        high_mean = np.mean(np.array(high_e), axis=0)

        ax_emb = high_mean - low_mean

        return ax_emb

    def score_terms_axis(self, terms, ax_emb):
        
        term_embs = [self.emb_model.get(term.lower()) for term in terms]
        term_embs = [x for x in term_embs if x is not None] #remove None
        if len(term_embs) == 0:
            return np.nan
        
        sims = cosine_similarity(np.array(term_embs), ax_emb.reshape(1, -1))
        return np.mean(sims.reshape(-1))

    ##################### avg of cosine sims ##############################

    def get_avg_cosine(self, terms, lex_words=None, lex_embs=None):

        # if (not lex_words) and (not lex_embs):
        #     print("List of lexicon terms OR lexicon term embeddings must be supplied as arguments")

        if lex_embs is None:
            if lex_words is None:
                # take the top 25 percentile of scoring terms
                scores = list(self.lexicon.values())
                lb = np.percentile(scores, 75)
                lex_words = [x for x,y in self.lexicon.items() if y>=lb]
            
            lex_embs = [self.emb_model.get(x.lower()) for x in lex_words]
            lex_embs = np.array([x for x in lex_embs if x is not None])
        
        score = np.nan # or 0?

        if len(lex_embs) == 0:
            return score

        term_embs = [self.emb_model.get(x.lower()) for x in terms]
        term_embs = np.array([x for x in term_embs if x is not None])

        if term_embs.shape[0] == 0:
            return score

        cosines = cosine_similarity(term_embs, lex_embs)

        return np.mean(np.mean(cosines, axis=1))


    ######################### main helper #################################
    
    def get_attr_scores(self, list_of_attrs, method=None, use_contrast=True):
        '''
        list_of_attrs: list of [string of attrs for story/sentence]
        method: one of avg, sim, axis
        use_contrast: if method is axis, specify bool value for use_contrast method
        returns a list: [score(x) for x in list_of_attrs]
        '''
        if not method:
            method = self.method
        
        if method not in ['avg', 'sim', 'axis']:
            print("Invalid method specific [avg/sim/axis]: defaulting to avg")
            method = 'avg'

        print("Method: ", method)

        scores = [np.nan for _ in list_of_attrs]

        if method == 'avg':
            scores = [self.get_lex_avg(x) for x in list_of_attrs]

        elif method == 'sim':
            scores = list(self.lexicon.values())
            lb = np.percentile(scores, 75)
            lex_words = [x for x,y in self.lexicon.items() if y>=lb]
            
            lex_embs = [self.emb_model.get(x.lower()) for x in lex_words]
            lex_embs = [x for x in lex_embs if x is not None]

            scores = [self.get_avg_cosine(x, lex_embs=lex_embs) for x in list_of_attrs]

        
        else:

            ax_emb = self.construct_axis(use_contrast)
            scores = [self.score_terms_axis(x, ax_emb) for x in list_of_attrs]

        return scores