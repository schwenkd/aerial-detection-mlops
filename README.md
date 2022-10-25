# Aerial-Detection-MLOps
<details>
    <summary> Step-1: Setup an EC2 box for Training </summary>

    - Instantiate a p3.2xlarge EC2 box (for data-parallel training get p3.8xlarge box) with the following AMI:
        Deep Learning AMI GPU PyTorch 1.12.1 (Amazon Linux 2) 20221005
    - Open ports 22, 80, 443, 800 from anywhere
    - AWS configure
    - conda activate pytorch
    - configure git if needed
    - initialize wandb

</details>
<details>
    <summary> Step-2: Convert VisDrone data to YOLO format:</summary>

    ## Step-2.1: Convert VisDrone DET data

        - git clone https://github.com/schwenkd/aerial-detection-mlops.git
        - cd aerial-detection-mlops
        - mkdir VisDrone
        - cd VisDrone
        - aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/DET/VisDrone2019-DET-train.zip VisDrone2019-DET-train.zip
        - aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/DET/VisDrone2019-DET-val.zip VisDrone2019-DET-val.zip
        - aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/DET/VisDrone2019-DET-train.zip VisDrone2019-DET-train.zip
        - unzip -d . VisDrone2019-DET-val.zip
        - unzip -d . VisDrone2019-DET-train.zip
        - unzip -d . VisDrone2019-DET-test-dev.zip
        - cd ..
        - python3 ./src/yolo_data_utils/convert_visdrone_DET_data_to_yolov7.py --output_image_size "(960, 544)"
        - aws s3 cp VisDrone2019-DET-YOLOv7.zip s3://aerial-detection-mlops4/data/visdrone/yolov7-data/DET/VisDrone2019-DET-YOLOv7.zip
        - You can cleanup the VisDrone directory by deleting all the zip files containing the raw data.

    ## Step-2.2: Convert VisDrone VID data

        - git clone https://github.com/schwenkd/aerial-detection-mlops.git
        - cd aerial-detection-mlops
        - mkdir VisDrone
        - cd VisDrone
        - aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/Video/VisDrone2019-VID-train.zip VisDrone2019-VID-train.zip
        - aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/Video/VisDrone2019-VID-test-dev.zip VisDrone2019-VID-test-dev.zip
        - aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/Video/VisDrone2019-VID-val.zip VisDrone2019-VID-val.zip
        - unzip -d . VisDrone2019-VID-train.zip
        - unzip -d . VisDrone2019-DEVIDT-test-dev.zip
        - unzip -d . VisDrone2019-VID-val.zip
        - cd ..        
        - python3 ./src/yolo_data_utils/convert_visdrone_VID_data_to_yolov7.py --output_image_size "(960, 544)"
        - aws s3 cp VisDrone2019-VID-YOLOv7.zip s3://aerial-detection-mlops4/data/visdrone/yolov7-data/Video/VisDrone2019-DET-YOLOv7.zip
        - You can cleanup the VisDrone directory by deleting all the zip files containing the raw data.

</details>

<details>
    <summary> Step-3: Do the training </summary>

        - git clone https://github.com/schwenkd/aerial-detection-mlops.git
        - cd aerial-detection-mlops
        - git clone https://github.com/WongKinYiu/yolov7.git
        - git clone https://github.com/ultralytics/yolov5.git
        - install requirements.txt libraries for both yolov7 and yolov5
        - mkdir VisDrone
        - cd VisDrone
        - aws s3 cp s3://aerial-detection-mlops4/data/visdrone/yolov7-data/DET/VisDrone2019-DET-YOLOv7.zip VisDrone2019-DET-YOLOv7.zip
        - unzip -d . VisDrone2019-DET-YOLOv7.zip
        - aws s3 cp s3://aerial-detection-mlops4/data/visdrone/yolov7-data/Video/VisDrone2019-VID-YOLOv7.zip VisDrone2019-VID-YOLOv7.zip
        - unzip -d . VisDrone2019-VID-YOLOv7.zip
        - cd ..
        - bash ./src/train/train_yolov7.sh


</details>
