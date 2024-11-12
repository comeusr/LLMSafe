#!/bin/bash

#SBATCH -A standby
#SBATCH --gpus-per-node=2
#SBATCH -C A100
#SBATCH --time=04:00:00
#SBATCH --job-name test
#SBATCH --output ./out.out
#SBATCH --error ./err.err

loss=kl
datasets=[kl] #[shp,hh,oasst]
model=llama7b_sft
exp_name=${loss}_${model}
cache=./data/models

cat <<EOF > nested_script.sh
#!/bin/bash
export TRANSFORMERS_CACHE=/depot/qfsong/LLM/project/hf_cache/hub
export HF_HOME=/depot/qfsong/LLM/project/hf_cache/
export HF_DATASETS_CACHE=/depot/qfsong/LLM/project/hf_cache/datasets
export WANDB_API_KEY=8e4a0bf8aa276a6a441763aab7441d43ed309205
cd /depot/qfsong/LLM/project/LLMSafe
python train.py loss=${loss} model=${model} datasets=${datasets} exp_name=${exp_name} mode=train ++cache_dir=${cache}
EOF
chmod +x nested_script.sh
apptainer exec --nv /depot/qfsong/LLM/project/halos.sif ./nested_script.sh
rm -rf nested_script.sh