# How to create tritonserver model artifacts from .pt file
# Create folder structure
mkdir -p triton-deploy/models/yolov7-visdrone-finetuned/1/
aws s3 cp s3://aerial-detection-mlops4/model/Visdrone/Yolov7/best/ae-yolov7-best-fp16-1x8x8.engine .
touch triton-deploy/models/yolov7-visdrone-finetuned/config.pbtxt
vim triton-deploy/models/yolov7-visdrone-finetuned/config.pbtxt
## write following content to the file
# name: "yolov7-visdrone-finetuned"
# platform: "tensorrt_plan"
# max_batch_size: 8
# dynamic_batching { }
#
# Place model
mv ae-yolov7-best-fp16-1x8x8.engine triton-deploy/models/yolov7/1/model.plan
aws s3 sync triton-deploy s3://aerial-detection-mlops4/model/Visdrone/Yolov7/triton-deploy
