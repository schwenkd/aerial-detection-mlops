# Aerial-Detection-MLOps
<details>
    <summary> Step-1: Setup an EC2 box for Training </summary>

- Instantiate a p3.2xlarge EC2 box (for data-parallel training get p3.8xlarge box) with the following AMI:
Deep Learning AMI GPU PyTorch 1.12.1 (Amazon Linux 2) 20221005
- Open ports 22, 80, 443, 8000-8002 from anywhere
- AWS configure
- source activate pytorch
- configure git if needed
- initialize wandb

</details>
<details>
    <summary> Step-2: Convert VisDrone data to YOLO format:</summary>

## Step-2.1: Convert VisDrone DET data

- git clone https://github.com/schwenkd/aerial-detection-mlops.git
- cd aerial-detection-mlops
- pip3 install -r yolov7/requirements.txt
- mkdir VisDrone
- cd VisDrone
- aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/DET/VisDrone2019-DET-train.zip VisDrone2019-DET-train.zip
- aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/DET/VisDrone2019-DET-val.zip VisDrone2019-DET-val.zip
- aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/DET/VisDrone2019-DET-test-dev.zip VisDrone2019-DET-test-dev.zip
- unzip -d . VisDrone2019-DET-val.zip
- unzip -d . VisDrone2019-DET-train.zip
- unzip -d . VisDrone2019-DET-test-dev.zip
- mkdir VisDrone2019-DET-val
- mv -r annotations images VisDrone2019-DET-val
- cd /home/ec2-user/aerial-detection-mlops
- mkdir -r VisDrone/VisDrone2019-VID-test-dev
- python3 ./src/yolo_data_utils/convert_visdrone_DET_data_to_yolov7.py --output_image_size "(960, 544)"
- ? is this supposed to be here?  aws s3 cp s3://aerial-detection-mlops4/data/visdrone/yolov7-data/DET/VisDrone2019-DET-YOLOv7.zip VisDrone2019-DET-YOLOv7.zip 
- You can cleanup the VisDrone directory by deleting all the zip files containing the raw data.

## Step-2.2: Convert VisDrone VID data

- cd VisDrone
- aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/Video/VisDrone2019-VID-train.zip VisDrone2019-VID-train.zip
- aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/Video/VisDrone2019-VID-test-dev.zip VisDrone2019-VID-test-dev.zip
- aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/Video/VisDrone2019-VID-val.zip VisDrone2019-VID-val.zip
- unzip -d . VisDrone2019-VID-train.zip
- unzip -d . VisDrone2019-VID-test-dev.zip
- unzip -d . VisDrone2019-VID-val.zip  
- mkdir -r VisDrone2019-VID-val/annotations
- mkdir -r VisDrone2019-VID-val/sequences
- cd ..
- python3 ./src/yolo_data_utils/convert_visdrone_VID_data_to_yolov7.py --output_image_size "(960, 544)"
- ? is this supposed to be here? aws s3 cp VisDrone2019-VID-YOLOv7.zip s3://aerial-detection-mlops4/data/visdrone/yolov7-data/Video/VisDrone2019-DET-YOLOv7.zip
- You can cleanup the VisDrone directory by deleting all the zip files containing the raw data.
</details>

<details>
    <summary> Step-3: Training YOLOv7 </summary>

- cd aerial-detection-mlops
- git clone https://github.com/ultralytics/yolov5.git
- pip3 install -r yolov5/requirements.txt 
- cd VisDrone
- ? is this supposed to be here? aws s3 cp s3://aerial-detection-mlops4/data/visdrone/yolov7-data/DET/VisDrone2019-DET-YOLOv7.zip VisDrone2019-DET-YOLOv7.zip
- ? is this supposed to be here? unzip -d . VisDrone2019-DET-YOLOv7.zip
- ? is this supposed to be here? aws s3 cp s3://aerial-detection-mlops4/data/visdrone/yolov7-data/Video/VisDrone2019-VID-YOLOv7.zip VisDrone2019-VID-YOLOv7.zip
- ? is this supposed to be here? unzip -d . VisDrone2019-VID-YOLOv7.zip
- ? is this supposed to be here? cd ..
- bash ./src/train/train_yolov7.sh
- Save the best model to s3:
	 aws s3 sync ./exp11 s3://aerial-detection-mlops4/model/Visdrone/Yolov7/<yyyyMMdd>/<run_name>


</details>
<details>
    <summary> Step-4: Inferencing </summary>

- cd yolov7
- aws s3 cp s3://aerial-detection-mlops4/model/Visdrone/Yolov7/20221026/exp11/weights/best.pt ae-yolov7-best.pt
- python3 detect.py --weights ae-yolov7-best.pt --conf 0.4 --img-size 640 --source ../9999938_00000_d_0000208.jpg
- python3 detect.py --weights ae-yolov7-best.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
### load the file to s3
- aws s3 cp runs/detect/exp2/9999938_00000_d_0000208.jpg s3://aerial-detection-mlops4/inferencing/test.jpg

</details>
<details>
    <summary> Step-4.1: Create YOLOv7 object-detection mp4 Video from VisDrone-VID-Sequence files </summary>

- go to aerial-detection-mlops directoru
- git pull https://github.com/schwenkd/aerial-detection-mlops.git
- create directories: "inferencing/video/input/" and "inferencing/video/output/"
- go to "aerial-detection-mlops/inferencing/video/input/" folder
- aws s3 cp  s3://aerial-detection-mlops4/data/visdrone/raw-data/Video/VisDrone2019-VID-test-challenge.zip VisDrone2019-VID-test-challenge.zip
- unzip -d . VisDrone2019-VID-test-challenge.zip
- go back to aerial-detection-mlops/yolov7 directory
- python3 ../src/yolo_data_utils/convert_image_sequences_to_video.py --image_sequence_folder ../inferencing/video/input/VisDrone2019-VID-test-challenge/sequences
   --output_mp4_video_folder ../inferencing/video/output --output_image_size "(960,544)" --fps 10
- go to  "aerial-detection-mlops/inferencing/video/output/" and execute the following command
- aws s3 cp uav0000006_06900_v.mp4 s3://deepak-mlops4-dev/capstone/deleteit/uav0000006_06900_v.mp4

</details>

<details>
    <summary> Step-5: Integrating DVC </summary>

- yum install pip
- pip3 install dvc
- dvc init
- dvc remote add yolov7_det_data -d s3://aerial-detection-mlops4/data/visdrone/yolov7-data/DET/VisDrone2019-DET-YOLOv7.zip
- dvc remote add raw_data_det_train s3://aerial-detection-mlops4/data/visdrone/raw-data/DET/VisDrone2019-DET-train.zip
- dvc remote add raw_data_det_val s3://aerial-detection-mlops4/data/visdrone/raw-data/DET/VisDrone2019-DET-val.zip
- dvc remote add raw_data_det_test-dev s3://aerial-detection-mlops4/data/visdrone/raw-data/DET/VisDrone2019-DET-test-dev.zip
- dvc remote add raw_data_det_test-challenge s3://aerial-detection-mlops4/data/visdrone/raw-data/DET/VisDrone2019-DET-test-challenge.zip
- dvc add .
- git commit -m "dvc init"

</details>

<details>
    <summary> Step-6: Lambda Function </summary>
    
- created Lambda function aerial-detection-mlops-lambda
- IAM role used is aerial-detection-mlops-lambda-role
- the function is triggered whenever we drop a file in s3://aerial-detection-mlops4/inferencing/photos/input folder
- this function will call a detection-service that will inturn call the triton server
</details>
