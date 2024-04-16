import os, re, sys, json, string, csv, gzip
import spacy
import pandas as pd
import logging
import numpy as np
from collections import Counter, defaultdict

sys.path.append('/h/vkpriya/Implicit_bias/story_analysis_code')

import kp_utils
import kp_funcs
# import process_story

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', required=True)
parser.add_argument('--outPath', required=True)
parser.add_argument('--lexPath', required=True)
parser.add_argument('--w2vPath', required=True)
parser.add_argument('--attrMethod', required=False, default='full')
parser.add_argument('--scoreMethod', required=False, default='avg')

'''
Storing outputs: store (prompt ID, story-within-prompt-ID, output score) in a CSV -- checkpointing implemented
'''
SAVE_EVERY = 50 # checkpoint frequency
# change for generated stories
# PROCESSED_PATH = '/ssd005/projects/story_bias/human_story_prompt/processed_stories'
# COMET_PATH = '/ssd005/projects/story_bias/human_story_prompt/comet_attrs'
PROCESSED_PATH = '/ssd005/projects/story_bias/human_story_prompt/gen_processed_stories'
COMET_PATH = '/ssd005/projects/story_bias/human_story_prompt/gen_comet_attrs'

def read_checkpoint(outPath):
    # checkpoint file
    # raise NotImplementedError
    ckpt_file = os.path.join(outPath, 'ckpt.csv')
    done_inds = set()
    if os.path.exists(ckpt_file): # store the prompt indices that are done.
        with open(ckpt_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                done_inds.add(int(line[0]))
    
    return done_inds

def write_checkpoint(outPath, done_inds):
    ckpt_file = os.path.join(outPath, 'ckpt.csv')
    with open(ckpt_file, 'w') as f:
        writer = csv.writer(f)
        for ind in done_inds:
            writer.writerow([ind])

def read_processed_stories(read_path=None):
    if read_path == None:
        read_path = PROCESSED_PATH 
    p_dict = {}

    subfs = []
    for fp in os.scandir(read_path):
        if os.path.isdir(fp):
            subfs.append(fp.path)
    if len(subfs) == 0:
        subfs.append(read_path)

    # pinds, tinds, targets = [], [], []

    for cf in subfs:
        for fp in os.scandir(cf):
            # kinda hardcoded
            if fp.name.split("_")[0] == 'prompts':
                with open(fp, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        si, ti, pp = row
                        si = int(si)
                        ti = int(ti)
                        pp = pp.strip()
                        # rows.append([si, ti, pp])
                        sd = p_dict.get(si, {})
                        sd[ti] = pp
                        p_dict[si] = sd
    return p_dict

def read_comet_attrs(read_path=COMET_PATH):
    c_dict = {}

    subfs = []
    for fp in os.scandir(read_path):
        if os.path.isdir(fp):
            subfs.append(fp.path)
    if len(subfs) == 0:
        subfs.append(read_path)

    for cf in subfs:
        for fp in os.scandir(cf):
            # kinda hardcoded
            if fp.name.split("_")[0] == 'prompts':
                with open(fp, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        si, ti, pp = row
                        si = int(si)
                        ti = int(ti)
                        pp = eval(pp)
                        # rows.append([si, ti, pp])
                        sd = c_dict.get(si, {})
                        sd[ti] = pp
                        c_dict[si] = sd
    return c_dict


def read_input_stories(dataPath, processed_story_path=None):
    # currently assumes the story_mapping.json form -- can be customized/standardized later
    # prompts (keys) are sorted -- this maps the prompt indices
    # stories for each prompt are also sorted (just to be safe) -- this maps the story-within-prompt-index

    with open(dataPath, 'r') as f:
        lines = json.loads(f.read().strip())
    
    prompts = list(lines.keys()) # dictionary is ordered as of python3.7
    targets = []
    for p in prompts:
        ts = lines[p]
        # process them later
        targets.append(ts)
    
    p_dict = read_processed_stories(processed_story_path)
    processed_targets = []
    processed_targets = []
    for pi in range(len(prompts)):
        tgs = []
        for ti in range(len(targets[pi])):
            ts = ''
            if (pi in p_dict) and (ti in p_dict[pi]):
                ts = p_dict[pi][ti]
            tgs.append(ts)
        processed_targets.append(tgs)

    return prompts, targets, processed_targets


def write_score_by_gender(pov_human, pi, ti, s, outPath):
    outF = outPath + "/" + pov_human + "_score.csv"
    with open(outF, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([pov_human, pi, ti, s])


class GetScore:
    
    def __init__(self, lex, w2v_model, attr_method, score_method):

        # self.preprocess=preprocess

        self.lex = lex
        self.w2v_model = w2v_model
        self.attr_method = attr_method
        self.score_method = score_method

        self.axis_emb = None
        self.lex_embs = None

        self.text_to_attr = kp_funcs.TextToAttrs(attr_method = attr_method)
        self.attr_to_score = kp_funcs.AttrsToScore(lexicon=lex, emb_model=w2v_model, method=score_method)

        if score_method == 'axis':
            if attr_method == 'comet':
                self.axis_emb = self.attr_to_score.construct_axis(comet=True)
            else:
                self.axis_emb = self.attr_to_score.construct_axis(comet=False)

        if score_method == 'sim':
            scores = list(self.lex.values())
            lb = np.percentile(scores, 75)
            lex_words = [x for x,y in self.lex.items() if y>=lb]
            
            lex_embs = [self.w2v_model.get(x.lower()) for x in lex_words]
            lex_embs = [x for x in lex_embs if x is not None]
            self.lex_embs = np.array(lex_embs)            

    def process_targets(self, targets):
        tgs = targets

        # tgs = [process_story.replaceAll(tg) for tg in targets]
        # assume input in processed
        
        tg_attrs = [self.text_to_attr.get_attrs(x) for x in tgs]

        scores = []

        if self.score_method == 'axis':
            scores = [self.attr_to_score.score_terms_axis(x, self.axis_emb) for x in tg_attrs]

        elif self.score_method == 'sim':
            scores = [self.attr_to_score.get_avg_cosine(x, lex_embs = self.lex_embs) for x in tg_attrs]

        elif self.score_method == 'comet_default':
            scores = [self.attr_to_score.comet_original_calculation(x) for x in tg_attrs]
            print("comet scores: ", scores)

        else: # average score
            scores = [self.attr_to_score.get_lex_avg(x) for x in tg_attrs]

        return scores
            
def main(dataPath, outPath, lexPath, w2vPath, attrMethod, scoreMethod):
    lex = kp_utils.read_lexicon(lexPath)
    w2v_model = kp_utils.WordEmbModel(w2vPath)

    prompts, targets, processed_targets = read_input_stories(dataPath)

    comet_dict = None
    if attrMethod == 'comet':
        comet_dict = read_comet_attrs()

    # pinds, tinds, targets = read_processed_stories(dataPath)
    print("Data: ", len(prompts), len(targets), len(processed_targets))
    done_inds = read_checkpoint(outPath) #pinds that are already done

    outF = os.path.join(outPath, 'out_scores.csv')
    print("Writing to: {}".format(outF))
    # errorF = os.path.join(outPath, "human_pov_errors.csv")
    #out header: [prompt_ind, target_ind, score, human_pov]

    scorer = GetScore(lex=lex, w2v_model=w2v_model, attr_method=attrMethod, score_method=scoreMethod)

    write_rows = []
    counter = 0

    skip_lst = [1266] # skipping because stories for this prompt is too long causing CUDA memory issues

    for pi, p in enumerate(prompts):

        if pi in done_inds:
            counter += 1
            continue
        
        # tgs = targets[pi]
        tgs = processed_targets[pi]
        
        if attrMethod == 'comet':
            # retrieve the attribute dicts
            ctgs = []
            for ti, _ in enumerate(tgs):
                ct = []
                if (pi in comet_dict) and (ti in comet_dict[ti]):
                    ct = comet_dict[pi][ti]
                ctgs.append(ct)
            tgs = ctgs
            
        scores = scorer.process_targets(tgs)

        # print("prompt num %s, scores: " % pi, scores)

        for ti, s in enumerate(scores):
            write_rows.append([pi, ti, s])
        
        counter += 1
        done_inds.add(pi)

        if (counter % SAVE_EVERY == 0) or (counter == len(prompts)):
            print("Prompt {}/{}".format(pi, len(prompts)))
            with open(outF, 'a+') as f:
                writer = csv.writer(f)
                for row in write_rows:
                    writer.writerow(row)

            write_checkpoint(outPath, done_inds)

            write_rows = []

    if len(write_rows) > 0:
        with open(outF, 'a+') as f:
            writer = csv.writer(f)
            for row in write_rows:
                writer.writerow(row)

        write_checkpoint(outPath, done_inds)

if __name__ == '__main__':
    args = parser.parse_args()
    dataPath = args.dataPath
    outPath = args.outPath
    if not os.path.exists(outPath):
        os.makedirs(outPath, exist_ok=True)
    lexPath = args.lexPath
    w2vPath = args.w2vPath
    attrMethod = args.attrMethod
    scoreMethod = args.scoreMethod
    # povPath = args.povPath

    main(dataPath, outPath, lexPath, w2vPath, attrMethod, scoreMethod)
