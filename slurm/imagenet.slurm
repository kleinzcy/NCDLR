#!/bin/bash
#SBATCH -J imagenet
#SBATCH -p critical
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=zhangchy2@shanghaitech.edu.cn
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:NVIDIATITANRTX:1
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBTACH --exclude=ai_gpu34

cd /public/home/zhangchy2/workdir/Project/TMLR2023/code
source activate uno

# python train.py --c config/ours/img50_100.yaml
# python LA.py --c config/ours/img50_100.yaml
# python LA.py --c config/ours/img50_50.yaml
# python train.py --c config/ours/img50_50_GT.yaml
# python LA.py --c config/ours/img50_50_GT.yaml
# python train.py --c config/ours/img50_100_GT.yaml
# python LA.py --c config/ours/img50_100_GT.yaml
python AutoNovel.py --c config/AutoNovel/img50_50.yaml
# python AutoNovel.py --c config/AutoNovel/img50_100.yaml