
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Set up python virtual environment using Python3.8 then install the following requirements.
```
python3 -m venv env
source env/bin/activate
```

This is an example of how to list things you need to use the software and how to install them.
* Prerequisites
  ```sh
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm  # install SpaCy models for English
  ```

### Installation

1. COMeT2

   Install COMeT2 according to https://github.com/vered1986/comet-commonsense


## Basic Data Cleaning

1. We are using https://paperswithcode.com/dataset/writingprompts WritingPrompts dataset, download, unzip and save the train.wp_source and train.wp_target in one place. Then run data_parsing/clean_data.py. This will create a train.wp_combined file, where each line follows the format of prompt + \<endprompt> + story

2. Parse human written stories from WritingPrompts dataset and save in json. Format: per each prompt, save all the stories for that prompt. Feel free to change the input directory input_dir to your own directory containing train.wp_combined file. 

      ```sh
      bash data_parsing/save_stories.sh
      ```

<!-- USAGE EXAMPLES -->
## Usage

1. Generate GPT written stories. 
```cd generate_story```
In this folder the ```generate_story_script.sh``` runs the ```generate_story.py```, which asks GPT to generate stories based on the same prompts as the human written stories in WritingPrompts dataset. Feel free to run ```generate_story_script.sh``` to replicate the story generation process. Our generated stories dataset is uploaded for reference, this dataset is used for analysis.

2. Getting the POV
Getting the POV of the stories and the prompts. Go into data_parsing/pov_utils.py. And run ```get_pov_by_story``` for each story, and ```get_pov_by_prompt``` for each prompt. Note the prompt index and story index directly follows from story_mapping.json. Since the dictionary is ordered in python 3.6, then the first story prompt of the dictionary has prompt index 0, and the first story of that prompt has index 0 and so on.

3. Calculating the dimension scores for intellect, appearance, power, valence, arousal, dominance.
We calculate dimension scores for each story, both for human written r/WritingPromps stories as well as for GPT generated stories. Run ```bash scripts/generate_scripts.sh``` to run ```story_analysis_code/kp_run_for_data.py```. This uses lexicons to calculate story scores based on the words in the story. For COMET attributes, run ```bash story_analysis_code/scripts/comet/generate_comet.sh```

4. Analyze data. See data_analysis folder.


## Data Pipeline Notes

Given: a list of stories $D$, a lexicon $L$ that associates word to score for a dimension $e$.

Expected: a list of scores S, where $S_{i}$ is the score associated with the $D_{i}$'s protagist for $e$.

## Files
- `kp_utils.py`: utils to read lexicons and w2v model
- `kp_funcs.py`: classes to convert from text to list of attributes and attributes to scores
- `kp_run_for_data.py`: wrapper to run for the writingPrompts dataset (includes checkpointing)

### Scripts to run
See folder `<ROOT>/scripts/kp/generate_scripts.sh` which will generate scripts for all specified combinations of emotions and hyperparameters.
Run `bash <ROOT>/scripts/kp/generate_scripts.sh`
Uncomment line #50 to submit and run.

## Hyperparameters:

1. Converting from text --> [attrs]

    1.1.  COMET 

    1.2.   spacy method A (**full**): extract all words from sentences where protagonist is the subject (`TextToAttrs: get_all_attributes`)
    
    1.3.   spacy method B (**sub**): extract all words attached to verbs whose subject is the protagonist (`TextToAttrs: filter_clauses`)
    
    1.4.   Baseline method (**all**): use all words in the text.

--


2. Converting from [attrs] --> score

    2.1. Method **avg**: ` (AttrsToScore: get_lex_avg) ` 
    ```
    mean([LEXICON(w)]) for w in attrs
    ```

    2.2. Method **sim** (this is the Understanding Implicit Bias approach for Appearance/Intelligence): (`AttrsToScore: get_avg_cosine`). Requires a word embedding model.
    ```
    mean([mean([cosine(emb(w), emb(l)) for l in LEXICON]) for w in attrs])
    ```

    2.3. Method **axis** (this is the Understanding Implicit Bias approach for Power. Requires at minimum a bi-modal lexicon, i.e, both positive and negative terms, or a real-valued lexicon): (`AttrsToScore: score_terms_axis`). Requires a word embedding model.
    ```
    mean([cosine(emb(w), axis_emb(L)) for w in attrs])
    ```


## Parameter Formats

1. **Lexicon (L)**: A file `<name>.csv` with two fields, `word` and `value`. If it a uni-modal lexicon (i.e, only a list of positive values), all the values must be 1. Method `axis` cannot be used to determine attribute scores. Should be passed to `AttrsToScore` as a dictionary.

2. **Word embedding model M**: Should be passed to the class `AttrsToScore` as a dictionary that associates words with embedding vectors (can be implemented as a custom class, must contain a `get` method that returns the embedding if present else a nan value)
