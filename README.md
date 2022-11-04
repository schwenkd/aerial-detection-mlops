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
        - aws s3 cp s3://aerial-detection-mlops4/data/visdrone/raw-data/DET/VisDrone2019-DET-test-dev.zip VisDrone2019-DET-test-dev.zip
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
    <summary> Step-3: Training YOLOv7 </summary>

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

<details>
    <summary> Step-7: Deploy on Nvidia Triton Inference Server </summary>

    ## Step 7.1: Export model to TensorRT via ONNX

        # Note - this should be performed on the platform that inference will occur on, because TensorRT conversion applies graph optimization using kernels specific to the GPU
        # bring in model weights
        # clone and switch to pytorch conda env, pip install -r requirements.txt and pip install onnx-simplifier
        # export model to onnx with grid, EfficientNMS plugin, and dynamic batch size (only img size changed from yolov7 instructions)
        - python export.py --weights ./ae-yolov7-best.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 960 960
        - if there is an error stating that the  onnx runtime module missing, pip install onnxruntime and/or onnxruntime-gpu
        # run appropriate nvidia TensorRT container from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
        - our AMI uses CUDA 11.6.1, so we use 22.02
        - docker run -it --rm --gpus=all nvcr.io/nvidia/tensorrt:22.02-py3
        # copy onnx model into container
        - docker cp ae-yolov7-best.onnx <container-id>:/workspace/
        # run trtexec to export model as TensorRT engine with FP16 precision, min batch 1, opt batch 8, and max batch 8
        - ./tensorrt/bin/trtexec --onnx=ae-yolov7-best.onnx --minShapes=images:1x3x960x960 --optShapes=images:8x3x960x960 --maxShapes=images:8x3x960x960 --fp16 --workspace=4096 --saveEngine=ae-yolov7-best-fp16-1x8x8.engine --timingCacheFile=timing.cache
        # test
        - ./tensorrt/bin/trtexec --loadEngine=ae-yolov7-best-fp16-1x8x8.engine
        # results will look like:
                    [11/07/2022-14:36:36] [I] === Performance summary ===
            [11/07/2022-14:36:36] [I] Throughput: 125.089 qps
            [11/07/2022-14:36:36] [I] Latency: min = 8.86523 ms, max = 9.88672 ms, mean = 9.21912 ms, median = 9.27188 ms, percentile(99%) = 9.76582 ms
            [11/07/2022-14:36:36] [I] End-to-End Host Latency: min = 8.93781 ms, max = 16.664 ms, mean = 15.7417 ms, median = 15.7519 ms, percentile(99%) = 16.513 ms
            [11/07/2022-14:36:36] [I] Enqueue Time: min = 1.20361 ms, max = 1.46375 ms, mean = 1.26678 ms, median = 1.26025 ms, percentile(99%) = 1.41699 ms
            [11/07/2022-14:36:36] [I] H2D Latency: min = 1.00757 ms, max = 1.90149 ms, mean = 1.25755 ms, median = 1.33453 ms, percentile(99%) = 1.61914 ms
            [11/07/2022-14:36:36] [I] GPU Compute Time: min = 7.82544 ms, max = 8.35481 ms, mean = 7.94364 ms, median = 7.93091 ms, percentile(99%) = 8.3118 ms
            [11/07/2022-14:36:36] [I] D2H Latency: min = 0.00976562 ms, max = 0.112427 ms, mean = 0.0179271 ms, median = 0.015625 ms, percentile(99%) = 0.0512238 ms
            [11/07/2022-14:36:36] [I] Total Host Walltime: 3.02184 s
            [11/07/2022-14:36:36] [I] Total GPU Compute Time: 3.00269 s
        # copy engine to host
        - docker cp <container-id>:/workspace/ae-yolov7-best-fp16-1x8x8.engine .

    ## Step 7.2 Run on Triton Inference Server

        # Create triton deploy folder structure
        # Move TensorRT model to triton directory and rename as "model.plan"
        - mv ae-yolov7-best-fp16-1x8x8.engine triton-deploy/models/ae-yolov7-best/1/model.plan
        # create and configure triton-deploy/models/yolov7/config.pbtxt
            name: "ae-yolov7-best"
            platform: "tensorrt_plan"
            max_batch_size: 8
            dynamic_batching { }
        # Start triton inference server (from yolov7 directory)
        - again make sure to use the correct triton server image for your version of CUDA and TensorRT. We are using CUDA 11.6.1 and TensorRT 8.2.3, so need Triton Server 22.02
        - docker run --gpus all --rm --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/triton-deploy/models:/models nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models --strict-model-config=false --log-verbose 1

