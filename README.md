
This repo borrows code from https://github.com/tenghaohuang/Uncover_implicit_bias and from https://github.com/ddemszky/textbook-analysis: 

**Story bias**

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
Download RocStory dataset from https://cs.rochester.edu/nlp/rocstories/

### Installation

1. COMeT2

   Install COMeT2 according to https://github.com/vered1986/comet-commonsense

### Note:

Any mentions of huan1287/Uncover_implicit_bias, feel free to switch for a folder under your alias {aliasname}/{foldername}.

## Basic Data Cleaning (Skip if having access to: /ssd005/projects/story_bias/human_story_prompt/story_mapping.json)

1. We are using https://paperswithcode.com/dataset/writingprompts WritingPrompts dataset, which can be found in /ssd005/projects/story_bias/human_story_prompt/writingPrompts.tar.gz. Unzip and save the train.wp_source and train.wp_target in one place (huan1287/Uncover_implicit_bias/data_cleaning is the easiest). Then run /h/huan1287/Uncover_implicit_bias/data_cleaning/clean_data.py. This will create a train.wp_combined file, where each line follows the format of prompt + \<endprompt> + story

2. Parse human written stories from WritingPrompts dataset and save in json. Format: per each prompt, save all the stories for that prompt.Feel free to change the input directory input_dir to your own directory containing train.wp_combined file. 

      ```sh
      sbatch save_stories_gpu.slrm
      ```

<!-- USAGE EXAMPLES -->
## Usage

Using the batch scripts, please change line 3-5 which activates my virtual environment into code that activates your virtual environment. Please first cd into the **scripts** folder, then run the following

1. 

NOTE: NO NEED TO RUN THIS if have access to /ssd005/projects/story_bias/human_story_prompt. All resulting data is stored there. Currently Kristin is fixing some data integrity issues.

Parse story_mapping.json into individual data files. Store all_texts to get lemmatized keywords of all stories to analyze the most common topics in the human written stories. For each prompt, store the topic_texts for each story in order to calculate topic prominence for each story and prompt. Additionally, find the POV of the story (first person unknown gender, third person female, third person male, others - unknown) as well as the number of characters in the story. All stored data is in the data folder. 

The POV of the prompt is separated into 4 groups: 'fm' if the prompt has stories with female and male protagonists. 'f' if stories only have female protagonists, 'm' if stories only have male protagonists. 'unknown' if stories only have first person/third person unknown gender protagonists.

      ```sh
      sbatch all_file_write_gpu.slrm
      ```

3. Generate stories using prompts with LLM (GPT-3.5) according to the gender of the prompts. See generate_story folder for code. First **organize_prompt.py** to organize prompt by gender, then **generate_story.py** to generate stories. NOTE: Right now the generate_story uses Kristin's OPENAI access key. Please change to your own OPENAI access key. 
Store the generated stories in /ssd005/projects/story_bias/human_story_prompt/gen_story. Right now we have generated approximately 1000 stories for female prompts, 945 stories for male prompts, 663 stories for fm_prompts, and 496 stories for unknown prompts. 

      ```sh
      sbatch gen_story_gpu.slrm
      ```

4. Analyze the human stories and the generated stories. First analyze the intellect, appearance, power by gender. 
See code in story_analysis_code folder. Create a story_analysis_data folder to store generated data. First **get_gen_stories.py** to retrieve the generated and human stories for each prompt and store by gender. Then **process_story.py** to mask the proagonist gender in the story. Then **generate_inferences.py** to use COMET commonsense reasoning to get protagonist attributes. These code can be run with: 
      ```sh
      sbatch process_story_gpu.slrm
      ```
Then to get lexicon score for intellect, appearance, and power, run **get_lexicon_scores.ipynb** to see graphs that show the distribution of these scores by gender.
To get valence, arousal, dominance scores, cd into scripts folder and run:
```sh
sbatch calc_VAD_gpu.slrm
```

In progress: calculating valence arousal dominance individually (see individual_connotation_COMET_NRC.ipynb)

To get pov of each story, we use kristin_pov.py run by /scripts/pov/generate_pov.sh