#!/bin/bash
# https://thelinuxcluster.com/2020/05/21/cant-execute-conda-activate-from-bash-script/
# source ~/conda/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate pytorch
# source ./src/train/train_config.sh
cd ./yolov5
num_freeze_layers=118
num_epochs=50
dataset_name="VisDrone2019-DET-YOLOv7"
batch_size=8
weights_filename="yolov5x.pt"
cfg_filename="yolov5s.yaml"
# python3 train.py --freeze 74 --batch 16 --epochs 5 --data ../VisDrone/VisDroneData-2019/data.yaml --weights 'yolov7-tiny.pt' --device 0 --cfg ./cfg/training/yolov7-tiny.yaml
# python3 train.py --freeze 102 --batch 16 --epochs 5 --data ../VisDrone/VisDroneData-2019/data.yaml --weights 'yolov7_training.pt' --device 0 --cfg ./cfg/training/yolov7.yaml
wget -nc https://github.com/ultralytics/yolov5/releases/download/v6.2/$weights_filename
#python3 train.py --img-size 960 --rect --batch $batch_size --epochs $num_epochs --data ../VisDrone/$dataset_name/data.yaml --weights $weights_filename  --upload_data val
python3 train.py --img-size 960 --rect --batch $batch_size --epochs $num_epochs --data ../VisDrone/$dataset_name/data.yaml --weights $weights_filename  --upload_data val