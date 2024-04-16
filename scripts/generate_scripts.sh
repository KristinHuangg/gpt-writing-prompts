#!/bin/bash
PROJECT_ROOT='/h/vkpriya/Implicit_bias'
ATTR_NAMES=("all" "full" "sub" "comet")
SCORE_NAMES=('avg' 'sim' 'axis')
EMOS=('arousal' 'dominance' 'valence' 'appearance' 'intellect' 'power')
LEX_ROOT=${PROJECT_ROOT}/lexicon_data/kp
W2V_PATH='/h/vkpriya/data/GoogleNews-vectors-negative300.bin'
scripts_dir=${PROJECT_ROOT}/scripts/kp/gen_all_combs
mkdir -p ${scripts_dir}
log_dir=${PROJECT_ROOT}/logs/kp/gen_sub_new
mkdir -p ${log_dir}
READ_PATH='/ssd005/projects/story_bias/human_story_prompt/gen_story_mapping.json'
OUT_PATH='/ssd005/projects/story_bias/human_story_prompt/gen_outputs/kp'
mkdir -p ${OUT_PATH}
partitions=('m2' 'm2' 'm2' 'm2' 'm2' 'm2' 'm2' 'm2' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3' 'm3') 
counter=0
for i in ${!EMOS[*]}; do
    for j in ${!ATTR_NAMES[*]}; do
        for k in ${!SCORE_NAMES[*]}; do
            partition=${partitions[$counter]}
            emo=${EMOS[$i]}
            lexp=${LEX_ROOT}/${emo}.csv
            attr=${ATTR_NAMES[$j]}
            score=${SCORE_NAMES[$k]}
            job_name=${emo}_${attr}_${score}
            echo ${job_name}
            echo ${partition}
            writeF=${OUT_PATH}/${emo}/${attr}/${score}
            # readf=${JSON_IN_PATH}
            mkdir -p ${writeF}
            echo "#!/bin/bash
. /etc/profile.d/lmod.sh
module use /pkgs/environment-modules/
echo values: --dataPath ${READ_PATH} --outPath ${writeF} --lexPath ${lexp} --w2vPath ${W2V_PATH} --attrMethod ${attr} --scoreMethod ${score}
python ${PROJECT_ROOT}/story_analysis_code/kp_run_for_data.py --dataPath ${READ_PATH} --outPath ${writeF} --lexPath ${lexp} --w2vPath ${W2V_PATH} --attrMethod ${attr} --scoreMethod ${score}
            " > ${scripts_dir}/${job_name}.sh
                echo "#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --open-mode=append
#SBATCH --qos=${partition}
#SBATCH --time=4:00:00
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
counter=$((counter + 1))
        done
    done
done