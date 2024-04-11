#!/bin/bash
. /etc/profile.d/lmod.sh
cd ..
cd ..
source 38env/bin/activate
cd ${REPO_DIR}
cd data_cleaning
echo Environment loaded
python save_stories.py --input_dir ${REPO_DIR}/train.wp_combined --output_dir ${REPO_DIR}/human_story_prompt
