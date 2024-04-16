import re
import os

name = "/ssd005/projects/story_bias/human_story_prompt/writingPrompts/train"
write_to_path = "/ssd005/projects/story_bias/human_story_prompt/writingPromptsCombined/train"
fp = open(name + ".wp_source") 
ft = open(name + ".wp_target") 

stories = ft.readlines()
prompts = fp.readlines()

combined = open(write_to_path + '.wp_combined', "w")
for i in range(len(stories)):
    prompt_new = re.sub('(\[|\()(\s[A-Za-z][A-Za-z]\s)(\]|\))', "", prompts[i]) # removes ( WP ), [ WP ] etc.
    cleaned_prompt = prompt_new.strip("\n").strip()
    combined.write(cleaned_prompt + " <endprompt> " + stories[i])

combined.close()

 
