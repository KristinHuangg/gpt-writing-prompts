#!/bin/bash
PROJECT_ROOT='/h/vkpriya/Implicit_bias'

scripts_dir=${PROJECT_ROOT}/story_analysis_code/scripts/comet/combs
mkdir -p ${scripts_dir}
log_dir=${PROJECT_ROOT}/logs/comet
mkdir -p ${log_dir}
# READ_PATH='/ssd005/projects/story_bias/human_story_prompt/processed_stories/'
READ_PATH='/ssd005/projects/story_bias/human_story_prompt/gen_processed_stories/'
# OUT_PATH='/ssd005/projects/story_bias/human_story_prompt/comet_attrs/'
OUT_PATH='/ssd005/projects/story_bias/human_story_prompt/gen_comet_attrs/'
mkdir -p ${OUT_PATH}
NUM=10000
STARTS=(0 10000 20000 30000 40000 50000 60000 70000 80000 90000)
partitions=('m2' 'm2' 'm2' 'm2' 'm2' 'm2' 'm2' 'm2' 'm' 'm' 'm' 'm' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3')
for i in ${!STARTS[*]}; do   
    st=${STARTS[$i]}
    partition=${partitions[$i]}
    job_name=prep_${st}
    echo ${job_name}
    writeF=${OUT_PATH}/start_${st}
    mkdir -p ${writeF}
    echo "#!/bin/bash
. /etc/profile.d/lmod.sh
module use /pkgs/environment-modules/
python ${PROJECT_ROOT}/story_analysis_code/comet_process_story.py --inputPath ${READ_PATH} --outputPath ${writeF} --startIndex ${st} --window ${NUM}
            " > ${scripts_dir}/${job_name}.sh
    echo "#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --open-mode=append
#SBATCH --qos=${partition}
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --signal=B:USR1@10
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_dir}/${job_name}_%j.out
handler()
{
    echo "function handler called at $(date)"
    sbatch ${scripts_dir}/${job_name}.slrm
}
trap handle SIGUSR1

bash ${scripts_dir}/${job_name}.sh" > $scripts_dir/${job_name}.slrm
sbatch $scripts_dir/${job_name}.slrm
done