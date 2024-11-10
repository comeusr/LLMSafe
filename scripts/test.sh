loss=kl
datasets=[kl] #[shp,hh,oasst]
model=llama7b_sft
exp_name=${loss}_${model}
cache=./data/models

export TRANSFORMERS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/hub
export HF_HOME=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/
export HF_DATASETS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/datasets
export WANDB_API_KEY=8e4a0bf8aa276a6a441763aab7441d43ed309205

python train.py loss=${loss} model=${model} datasets=${datasets} exp_name=${exp_name} mode=train ++cache_dir=${cache}