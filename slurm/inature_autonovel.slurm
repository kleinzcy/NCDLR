#!/bin/bash
#SBATCH -J i_auto
#SBATCH -p ShangHAI
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=zhangchy2@shanghaitech.edu.cn
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBTACH --exclude=ai_gpu34

cd /public/home/zhangchy2/workdir/Project/TMLR2023/code
source activate uno
CUDA_VISIBLE_DEVICES=0 python AutoNovel.py --c config/AutoNovel/inaturalist2k.yaml
# CUDA_VISIBLE_DEVICES=0 python LA.py --c config/AutoNovel/inaturalist.yaml
# CUDA_VISIBLE_DEVICES=0 python AutoNovel.py --c config/AutoNovel/inaturalist2k.yaml
# CUDA_VISIBLE_DEVICES=0 python LA.py --c config/AutoNovel/inaturalist2k.yaml