#!/bin/bash
PROJECT_ROOT='/story_bias'
code_dir='/story_bias/generate_story'
scripts_dir=${code_dir}/scripts
mkdir -p ${scripts_dir}
log_dir='/scratch/ssd004/scratch/huan1287/gen_story_log'
mkdir -p ${log_dir}
partitions=('normal' 'normal' 'm' 'm' 'm' 'm')
counter=0
for i in ${!prompt_types[*]}; do
    partition=${partitions[$counter]}
    prompt_type=${prompt_types[$i]}
    job_name=${prompt_type}
    echo ${job_name}
    echo ${partition}
    echo "#!/bin/bash
. /etc/profile.d/lmod.sh
cd
source 38env/bin/activate
cd Story_bias
cd generate_story
echo Environment loaded
python ${code_dir}/generate_story.py --input ${INPUT_DIR} --output ${OUTPUT_DIR}
    " > ${scripts_dir}/${job_name}.sh
        echo "#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --open-mode=append
#SBATCH --qos=${partition}
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --signal=B:USR1@10
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_dir}/job_%j.out
handler()
{
    echo "function handler called at $(date)"
    sbatch ${scripts_dir}/${job_name}.slrm
}
trap handle SIGUSR1

bash ${scripts_dir}/${job_name}.sh" > $scripts_dir/${job_name}.slrm
sbatch $scripts_dir/${job_name}.slrm
counter=$((counter + 1))
done