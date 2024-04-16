
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
We calculate dimension scores for each story, both for human written r/WritingPromps stories as well as for GPT generated stories. Run ```bash scripts/generate_scripts.sh``` to run ```story_analysis_code/kp_run_for_data.py```. This uses lexicons to calculate story scores based on the words in the story. 
