# CUDA_VISIBLE_DEVICES=0 /home/lies_mlp/miniconda3/envs/liesmlp/bin/python main.py

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 main.py
