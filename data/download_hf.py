# set the mirror if necessary
# export HF_ENDPOINT=https://hf-mirror.com

from huggingface_hub import snapshot_download, hf_hub_download

snapshot_download(repo_id='GD-ML/UniVG-R1-data', # Michael4933/MIG-Bench
                  repo_type='dataset',
                  local_dir='/home/tajamul/UniVG-R1/eval/MIG_bench',
                  force_download=False,
                  max_workers=32,
                )