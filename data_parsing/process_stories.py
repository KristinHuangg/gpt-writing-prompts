# anonymize the story's protagonist

import nltk
from nltk.tag.stanford import StanfordNERTagger
from allennlp.predictors.predictor import Predictor
from comet2.comet_model import PretrainedCometModel
import spacy
import os, csv

import json
import argparse


predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",cuda_device = 0)
predictor._model = predictor._model.cuda()
print(predictor.cuda_device)
print("predictor loaded")

nlp = spacy.load('en_core_web_lg')

parser = argparse.ArgumentParser()

parser.add_argument('--inputPath', required=True, help="File containing the input data.")
parser.add_argument('--outputPath', required=True, help="Directory to write output files.")
parser.add_argument('--startIndex', required=False, default=0, type=int, help='start index of prompt')
parser.add_argument('--window', required=False, default=10000, type=int, help='number of prompts to process')


# comet_model = PretrainedCometModel(device=0, model_name_or_path='/h/vkpriya/comet-commonsense/comet-data/models/atomic_pretrained_model_openai-gpt/')

print("everything loaded")

def read_input_stories(dataPath, start_index, length):
    # currently assumes the story_mapping.json form -- can be customized/standardized later
    # prompts (keys) are sorted -- this maps the prompt indices
    # stories for each prompt are also sorted (just to be safe) -- this maps the story-within-prompt-index

    with open(dataPath, 'r') as f:
        lines = json.loads(f.read().strip())
    
    all_prompts = lines.keys() # dictionary is ordered as of python3.7

    prompts = []
    targets = []
    counter = 0
    for i,p in enumerate(all_prompts):
        if i<start_index:
            continue

        prompts.append(p)
        ts = lines[p]
        # process them later
        targets.append(ts)
        counter += 1
        if counter == length:
            break
    
    return prompts, targets

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


def getPronReplaceGen(para):

    output = predictor.predict(
        document=para
    )
    rt = []
    characters = [i[0] for i in getCharacters(para)]
    female_pron = ["She", "she", "Her", "her"]
    male_pron = ["He", "he", "him", "Him"]
    unresolved_pron = ["I", "Me", "me"]
    characters = characters + female_pron + male_pron + unresolved_pron

    for i in output['clusters']:
        for j in i:

            ent = " ".join(output['document'][j[0]:j[1] + 1])

            if (ent in characters):
                rt.append(i)
                break

    return rt


def getCharacters(sentence):
    doc1 = nlp(sentence)
    entities = []
    for ent in doc1.ents:
        entities.append((ent.text, ent.label_))
    characters = [i for i in entities if (i[1] == 'PERSON')]
    return set(characters)


def isPron(pt, li):
    for num, tup in enumerate(li):
        if (tup[0] <= pt and pt < tup[1]):
            return tup[1], True
        if (tup[0] == pt):
            return pt, True

    return pt, False


def replaceAll(sentence):
    if len(sentence.split())>500:
        sentence = " ".join(sentence.split()[:500])

    sentence = sentence.replace("<newline>", "\n")
    sentence = sentence.replace("< newline >", "\n")
    
    doc = nlp(sentence)
    token_dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    clusters = getPronReplaceGen(sentence)
    init = 65

    curpos = 0
    snt = []
    while curpos < len(token_dependencies):
        flags = []
        positions = []
        characters = []
        for num, group in enumerate(clusters):
            (curpos, flag) = isPron(curpos, group)
            flags.append(flag)

            if flag:
                characters.append("Protagonist" + chr(init + num))
                positions.append(curpos)

        if (len(positions) > 0):
            curpos = max(positions)
            character = characters[positions.index(curpos)]
        if True in flags:
            #                 print(token_dependencies[curpos])
            if (token_dependencies[curpos][1] == 'poss'):
                #                     if(token_dependencies[curpos+1][1] == 'case'):
                #                         curpos += 1
                snt.append(character + "'s")
            elif (token_dependencies[curpos][1] == 'case'):
                snt.append(character + "'s")
            else:
                snt.append(character)
        else:
            snt.append(token_dependencies[curpos][0])
        curpos += 1

    ans = " ".join(snt) + '\n'

    return ans

def writeFile(file_name, content):
    a = open(file_name, 'a')
    a.write(content)
    a.close()

SAVE_EVERY = 50
def main(ip_path, op_path, start_ind, num_prompts):
    prompts, targets = read_input_stories(ip_path, start_ind, num_prompts)

    print(len(prompts), len(targets))

    end_ind = start_ind + num_prompts - 1 # should be start_ind + len(prompts) - 1
    op_f = os.path.join(op_path, 'prompts_{}_{}.csv'.format(start_ind, end_ind))

    os.makedirs(op_path, exist_ok=True)

    done_inds = read_checkpoint(op_path)

    inds = list(range(start_ind, start_ind + len(prompts) + 1))
    # counter = len(done_inds)
    counter = 0
    write_rows = []

    for pind, pr, tgs in zip(inds, prompts, targets):
        if pind in done_inds:
            counter += 1
            continue

        p_tgs = []
        for t in tgs:
            try:
                pt = replaceAll(t)

            except:
                pt = ''
            p_tgs.append(pt)

        for ti, t in enumerate(p_tgs):
            # prompt index, target index, processed story
            write_rows.append([pind, ti, t])

        counter += 1
        done_inds.add(pind)

        if (counter % SAVE_EVERY == 0) or counter == len(prompts):
            with open(op_f, 'a+') as f:
                writer = csv.writer(f)
                for row in write_rows:
                    writer.writerow(row)
            
            write_checkpoint(op_path, done_inds)

            write_rows = []
    
    if len(write_rows) > 0:
        with open(op_f, 'a+') as f:
            writer = csv.writer(f)
            for row in write_rows:
                writer.writerow(row)
        
        write_checkpoint(op_path, done_inds)

if __name__ == '__main__':
    args = parser.parse_args()
    ip_path = args.inputPath
    op_path = args.outputPath
    start_ind = int(args.startIndex)
    num_prompts = int(args.window)

    main(ip_path, op_path, start_ind, num_prompts)

