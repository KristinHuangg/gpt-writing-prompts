import openai
import os
import argparse
import time
import json, csv
import pandas as pd
openai.api_key = os.getenv("OPENAI_API_KEY")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

MODEL = "gpt-3.5-turbo"

parser = argparse.ArgumentParser()

parser.add_argument('--input', required=True, help="Directory and file containing the input data.")
parser.add_argument('--output', required=True, help="Directory containing the output files.")
parser.add_argument('--startIndex', required=False, default=0, type=int, help='start index of prompt')
parser.add_argument('--window', required=False, default=10000, type=int, help='number of prompts to process')

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


@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.ServiceUnavailableError, openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(15))
def get_gpt_response(prompt, role, model=MODEL, temp=0.95):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
    )
    new_story = response['choices'][0]['message']['content']
    return new_story

SAVE_EVERY = 10
def main(ip_path, op_path, start_ind, num_prompts):
    
    with open(ip_path, 'r') as f:
        lines = json.loads(f.read().strip())
    
    end_ind = start_ind + num_prompts

    all_prompts = list(lines.keys()) # dictionary is ordered as of python3.7

    prompts = all_prompts[start_ind : end_ind]
    inds = [i for i in range(start_ind, end_ind)]
    inds = inds[:len(prompts)]

    assert len(inds) == len(prompts)

    os.makedirs(op_path, exist_ok=True)
    outF = os.path.join(op_path, 'gen_stories.csv')
    print("Writing to: {}".format(outF))
    
    done_inds = read_checkpoint(op_path)
    prompt_prefix = "You're writing a Reddit story and you want other reddit users to like and upvote your story."
    instr = "Write a 400 words story for the following prompt: "
    write_rows = []
    counter = 0

    for pind, p in zip(inds, prompts):
        if pind in done_inds:
            counter += 1
            continue
        
        full_prompt = instr + p.strip()
        gen_story = get_gpt_response(full_prompt, prompt_prefix)

        write_rows.append([pind, prompt_prefix, full_prompt, gen_story])

        counter += 1
        done_inds.add(pind)

        if (counter % SAVE_EVERY == 0) or (counter == len(prompts)):
            print("Prompt {}/{}".format(pind, inds[-1]))
            with open(outF, 'a+') as f:
                writer = csv.writer(f)
                for row in write_rows:
                    writer.writerow(row)

            write_checkpoint(op_path, done_inds)

            write_rows = []
        
    if len(write_rows) > 0:
        with open(outF, 'a+') as f:
            writer = csv.writer(f)
            for row in write_rows:
                writer.writerow(row)

        write_checkpoint(op_path, done_inds)


if __name__=='__main__':
    args = parser.parse_args()

    ip_path = args.input
    op_path = args.output
    start_ind = int(args.startIndex)
    num_prompts = int(args.window)

    main(ip_path, op_path, start_ind, num_prompts)




