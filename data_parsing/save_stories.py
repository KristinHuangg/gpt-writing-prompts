import argparse
import json
parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', required=True, help="Directory of input text files.")
parser.add_argument('--human_written', default=True, type=bool, help="Analyzing human stories versus gpt stories")
parser.add_argument("--output_dir",
                    help=("output directory for intermediate data"),
                    type=str)
args = parser.parse_args()


def clean_human_dataset(line):
    if not line.isspace():
        tmp = line.split(" <endprompt> ")
        prompt = tmp[0]
        # if 'Two identical twins switch bodies .' in prompt:
        #     prompt = 'Two identical twins switch bodies .'
        human_story = tmp[1]
    return prompt, human_story

def create_init_mapping(human_written=args.human_written):
    count = 0
    f = open(args.input_dir, "r")
    mapping = {}
    lines = f.readlines()
    for i in lines:
        print(count)
        count+= 1
        if human_written == True:
            prompt, human_story = clean_human_dataset(i)
            prompt = prompt.rstrip()
            human_story = human_story.rstrip()
            if prompt in mapping:
                mapping[prompt].append(human_story)
            else:
                mapping[prompt] = [human_story]
    return mapping

def main():
    mapping = create_init_mapping()
    json_object = json.dumps(mapping, indent = 4) 
    with open("%s/story_mapping.json" % (args.output_dir), "w") as outfile:
        outfile.write(json_object)
    print("finished writing")

if __name__ == "__main__":
    main()