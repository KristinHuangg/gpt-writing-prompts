# GPT-WritingPrompts: A Comparative Analysis of Character Portrayal in Short Stories

Data and code for the [paper]().

## Getting Started

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

1. COMeT2 for commensense attributes

   Install COMeT2 according to https://github.com/vered1986/comet-commonsense

2. AllenNLP SpanBERT model for coreference resolution
    
    Install the [AllenNLP library](https://github.com/allenai/allennlp). (Can be replaced with a coreference resolution module of your choice.)

## Data

Download the [GPT-WritingPrompts dataset from HuggingFace](https://huggingface.co/datasets/vkpriya/GPT-WritingPrompts) and store in the `data/` subfolder of this repo.

### Prompts and Stories
The files `data/human_wp_stories.json` and `data/gpt_wp_stories.json` are JSON-formatted files containing the human-written (from the [WritingPrompts]() dataset) and gpt3.5-generated stories for the prompts from the training subset of the [WritingPrompts]() dataset.

### PoV information and processed stories
In the `data/meta_info` folder, there are two gzipped CSV files that contain PoV information and the protagonist-replaced (formatted for extracting protagonist attributes --  see Code subsection below) story for each human-written and machine-generated story.

### Story scores
The `outputs/` folder contains the computed story scores for each dimension, for each attribute extraction and attribute scoring method, for both the human-written and gpt-generated story subsets.

### Lexicons
We primarily use lexicons to infer word-level scores along each dimension of interest. We look at the following six dimensions: valence, arousal, dominance (together VAD), power (quite similar to dominance, kept for posterity w.r.t prev work), appearance, intellect. Lexicons we use can be found in `lexicon_data`.

- VAD are represented with a bipolar scale and real-valued scores between 0 and 1 (0 is lowest V/A/D, 1 is highest). We use the [NRC-VAD lexicons](http://saifmohammad.com/WebPages/nrc-vad.html). We only keep terms with scores >=0.67 (high scoring) and <=0.33 (low scoring), removing the neutral middle section.

- Fast et al. (2016b) create a crowdsourced resource of terms representing various stereotypes associated with characters. This list can be found in the associated [Empath library](https://github.com/Ejhfast/empath-client/blob/master/empath/data/categories.tsv).

- Power is a bipolar, boolean scale (0 for low power, 1 for high). High power terms are those associated with the term `powerful` in the Empath lexicon. Low power terms are those associated with `weak` in the Empath lexicon.

- Appearance and Intellect are uni-polar indicators of association. Both `intelligent` and `stupid` are therefore scored 1 on the lexicon, because they indicate that intellect-associated terms are being used. Same for appearance. 

- Based on prior work, we take terms associated with the seed words `["intellectual", "knowledgeable","intelligent", "educated", "skillful", "strategic", "scientific"]` in the Empath lexicon, and also add in close antonyms, to create the set of terms in `lexicon_data/intellect.csv`. (not manually validated, could be improved)

- For appearance, we use the seed words `["beautiful","sexual"]`, and again add in close (measured with cosine similarity of word2vec embeddings) antonyms. Antonyms are found using wordnet.


## Code
There are several parts to the pipeline. We describe them sequentially below.

### Prep work
- **Formatting the story data**: We re-formatted the original training set of the [WritingPrompts]() dataset into a `json` file for easier processing. This file is created using the code in `data_parsing/save_stories.py`.

- **Generating GPT-3.5 stories**: We prompt the GPT-3.5 model (can replace with the version of your choice) to generate 500-word stories given the initial prompt. Find our code to do this in `generate_story/generate_story.py`.

- **Inferring point-of-view**: We infer the point-of-view (first or second person, male/female/other third person) of each story using the SpanBERT coreference resolution model from AllenNLP. The most frequently-mentioned character entity is termed the *protagonist*. Code for this step is in `data_parsing/pov_utils.py`.

- **Replacing protagonist tokens**: We next replace each protagonist-coreferent token in each story (i.e, all tokens that refer to the protagonist, inferred using the SpanBERT model) with the special `protagonistA` token. This makes it easy to then extract attributes associated with the protagonist. Code to do this is in `data_parsing/process_stories.py`. We provide these processed stories in the `data/meta_info` files.

### Main method
- **Extracting attributes**: We now extract the protagonist-attributes that we want to score for each story. As described in the story, we use two main methods: dependency relations using SpaCy (`sub`), and commonsense inferece with COMeT (`comet`). In `story_analysis/attr_score_funcs.py`, you can find the class `TextToAttrs` that implements these methods.

- **Scoring attributes**: We then score each attribute token along each of our dimensions of interest (quantified with the lexicons in `lexicon_data`) using three methods: directly look it up in the lexicon (`avg`), use word2vec-based cosine similarities with lexicon terms (`sim`), or axis projection using the lexicon terms (`axis`). The code for this is in the `AttrsToScore` class in `story_analysis/attr_score_funcs.py`. The resulting scores for our dataset are provided in `outputs/story_scores`.

   - We evaluate if `sim` or `axis` is better at score estimation using the VAD lexicon to create train-test splits. The code for this eval is in `eval_scoring_methods.py`. `axis` is overall better. 

## Analysis
The `outputs-results-analysis.ipynb` notebook contains code to then go through these outputs and replicate the figures and tables in the paper.

## Contact
Contact the authors with any questions:
1. Kristin (Xi Yu) Huang, xiyu.huang@mail.utoronto.ca
2. Krishnapriya Vishnubhotla, vkpriya@cs.toronto.edu
3. Frank Rudzicz, frank@dal.ca
