#!/bin/bash
# https://thelinuxcluster.com/2020/05/21/cant-execute-conda-activate-from-bash-script/
# source ~/conda/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate pytorch
# source ./src/train/train_config.sh
cd ./yolov7
# num_freeze_layers=118 # corresponds to yolov7x
# num_freeze_layers=59 # corresponds to yolov7x backbone
#num_freeze_layers=102 # corresponds to yolov7
num_freeze_layers=51 # corresponds to yolov7 backbone
num_epochs=300
dataset_name="VisDrone2019-DET-YOLOv7"
batch_size=20
# weights_filename="yolov7x_training.pt"
# cfg_filename="yolov7x.yaml"
weights_filename="yolov7_training.pt"
cfg_filename="yolov7.yaml"
hyp_filename="yolov7_aerial_detection_hyp.yaml"
# python3 train.py --freeze 74 --batch 16 --epochs 5 --data ../VisDrone/VisDroneData-2019/data.yaml --weights 'yolov7-tiny.pt' --device 0 --cfg ./cfg/training/yolov7-tiny.yaml
# python3 train.py --freeze 102 --batch 16 --epochs 5 --data ../VisDrone/VisDroneData-2019/data.yaml --weights 'yolov7_training.pt' --device 0 --cfg ./cfg/training/yolov7.yaml
wget -nc https://github.com/WongKinYiu/yolov7/releases/download/v0.1/$weights_filename
#python3 train.py --adam --img-size 960 960 --batch $batch_size --epochs $num_epochs --freeze $num_freeze_layers --data ../VisDrone/$dataset_name/data.yaml --weights $weights_filename --device 0 --cfg ./cfg/training/$cfg_filename --hyp ../src/train/$hyp_filename
#python3 -m torch.distributed.launch --nproc_per_node 4 train.py --adam --img-size 960 960 --batch $batch_size --freeze $num_freeze_layers --epochs $num_epochs --data ../VisDrone/$dataset_name/data.yaml --weights $weights_filename --cfg ./cfg/training/$cfg_filename --hyp ../src/train/$hyp_filename
python3 -m torch.distributed.launch --nproc_per_node 4 train.py --adam --img-size 960 960 --batch $batch_size --epochs $num_epochs --data ../VisDrone/$dataset_name/data.yaml --weights $weights_filename --cfg ./cfg/training/$cfg_filename --hyp ../src/train/$hyp_filename
