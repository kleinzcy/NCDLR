#!/bin/bash
#SBATCH -J cifar
#SBATCH -p critical
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=zhangchy2@shanghaitech.edu.cn
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBTACH --exclude=ai_gpu34

cd /public/home/zhangchy2/workdir/Project/TMLR2023/code
source activate uno

# python train.py --c config/ours/cifar50_50_GT.yaml
python train.py --c config/ours/cifar50_100_GT.yaml

# python UNO.py --c config/UNO/cifar50_50.yaml
# python UNO.py --c config/UNO/cifar50_100.yaml

# python AutoNovel.py --c config/AutoNovel/cifar50_50.yaml
# python LA.py --c config/AutoNovel/cifar50_50.yaml
# python AutoNovel.py --c config/AutoNovel/cifar50_50_crt.yaml

# python UNO.py --c config/ours/cifar50_100.yaml
