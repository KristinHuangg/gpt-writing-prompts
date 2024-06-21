import os, re, sys, json, string, gzip, csv
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd

import attr_score_funcs
import utils

import scipy.stats as stats

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dim', help='name of the dimension')
parser.add_argument('--lex_path', help='path to the dimension lexicon')
parser.add_argument('--emb_path', help='path to the word embedding model')
parser.add_argument('--out_dir', help='path to the output folder')
parser.add_argument('--method', help='scoring method to evaluate')

def get_train_test_dists(lexicon, test_ratio = 0.3):
    words = [k for k in lexicon]
    scores = [lexicon[k] for k in words]

    train_x, test_x, train_y, test_y = train_test_split(words, scores, shuffle=True, test_size=test_ratio, random_state=42)

    return train_x, train_y, test_x, test_y


class ScoreEvaluator:

    def __init__(self, lexicon, emb_model, method, num_trials = 5):

        assert method in ['sim', 'axis_with_con', 'axis_no_con'], "invalid method (not in [sim, axis_with_con, axis_no_con])"

        self.lexicon = lexicon
        self.emb_model = emb_model
        self.method = method
        self.num_trials = num_trials

    def eval_sim(self, train_x, train_y, test_x, test_y):

        attr_to_score = attr_score_funcs.AttrsToScore(self.lexicon, self.emb_model, 'sim')
        lb = np.percentile(train_y, 75) # 75th percentile
        lex_words = [x for x,y in zip(train_x, train_y) if y>=lb]
        print("Extracted {} lexicon terms at a threshold of {}".format(len(lex_words), lb))

        lex_embs = [self.emb_model.get(x.lower()) for x in lex_words]
        lex_embs = [x for x in lex_embs if x is not None]

        test_pred = [attr_to_score.get_avg_cosine([x], lex_embs=lex_embs) for x in test_x]

        #spearman correlation (ranking)
        sp_corr, sp_p = stats.spearmanr(test_y, test_pred)

        return sp_corr, sp_p

    def eval_axis(self, train_x, train_y, test_x, test_y, use_contrast):
        attr_to_score = attr_score_funcs.AttrsToScore(self.lexicon, self.emb_model, 'axis')
        lb = np.percentile(train_y, 25)
        ub  = np.percentile(train_y, 75)

        low_t = [k for k,v in zip(train_x, train_y) if v <= lb]
        low_t = [k for k in low_t if k in attr_to_score.emb_model.w2v_model]
        high_t = [k for k,v in zip(train_x, train_y) if v>=ub]
        high_t = [k for k in high_t if k in attr_to_score.emb_model.w2v_model]

        print("Found {} low and {} high terms with thresholds {} and {}".format(len(low_t), len(high_t), lb, ub))

        low_e = [attr_to_score.emb_model.get(w) for w in low_t]
        high_e = [attr_to_score.emb_model.get(w) for w in high_t]

        if use_contrast:
            sim_matrix = attr_to_score.get_sim_matrix(low_e, high_e)

            contrast_pairs = attr_to_score.get_contrast_pairs([low_t, high_t], sim_matrix)
            print("Found {} contrast pairs".format(len(contrast_pairs)))
            print(contrast_pairs[:5])
            print()

            low_pole = [x[0] for x in contrast_pairs]
            high_pole = [x[1] for x in contrast_pairs]
            low_e = [attr_to_score.emb_model.get(w) for w in low_pole]
            high_e = [attr_to_score.emb_model.get(w) for w in high_pole]

        low_mean = np.mean(np.array(low_e), axis=0)
        high_mean = np.mean(np.array(high_e), axis=0)

        ax_emb = high_mean - low_mean

        test_pred = [attr_to_score.score_terms_axis([x], ax_emb) for x in test_x]
        sp_corr, sp_p = stats.spearmanr(test_y, test_pred)

        return sp_corr, sp_p


    def test_scoring_method(self):

        sp_corrs = []
        sp_ps = []
        for t_num in range(self.num_trials):
            train_x, train_y, test_x, test_y = get_train_test_dists(self.lexicon)

            train_keep_inds = [i for i,x in enumerate(train_x) if x in self.emb_model.w2v_model]
            test_keep_inds = [i for i,x in enumerate(test_x) if x in self.emb_model.w2v_model]

            train_x = [train_x[i] for i in train_keep_inds]
            train_y = [train_y[i] for i in train_keep_inds]
            test_x = [test_x[i] for i in test_keep_inds]
            test_y = [test_y[i] for i in test_keep_inds]

            if self.method == 'sim':
                sp_c, sp_p = self.eval_sim(train_x, train_y, test_x, test_y)
            elif self.method == 'axis_with_con':
                sp_c, sp_p = self.eval_axis(train_x, train_y, test_x, test_y, use_contrast=True)
            else:
                sp_c, sp_p = self.eval_axis(train_x, train_y, test_x, test_y, use_contrast=False)

            sp_corrs.append(sp_c)
            sp_ps.append(sp_p)
        
        return sp_corrs, sp_ps

def main(dim, lex_path, emb_path, save_dir, method):
    lex = utils.read_lexicon(lex_path)
    w2v_model = utils.WordEmbModel(emb_path)

    evaluator = ScoreEvaluator(lex, w2v_model, method)

    sp_corrs, sp_ps = evaluator.test_scoring_method()

    df = pd.DataFrame({
        'sp_corr': sp_corrs,
        'sp_p': sp_ps
    })

    df['dim'] = dim
    df['method'] = method

    os.makedirs(save_dir, exist_ok=True)

    savef = os.path.join(save_dir, "-".join([dim, method]))

    df.to_csv(savef, index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    dim = args.dim
    lex_path = args.lex_path
    emb_path = args.emb_path
    save_dir = args.out_dir
    method = args.method

    main(dim, lex_path, emb_path, save_dir, method)
