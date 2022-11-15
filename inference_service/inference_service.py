import argparse
from fastapi import FastAPI, File, UploadFile
from triton_client import TritonClient
import logging
from urllib.parse import unquote
import boto3
import os
import time

my_triton_url = 'triton:8001'
my_model = 'yolov7-visdrone-finetuned'
# setup loggers
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='Aerial Detection Inference Service')
tmp_file_folder_input = "./tmp_data/input"
tmp_output_img_folder = "./tmp_data/output/image"
tmp_output_lbl_folder = "./tmp_data/output/label"

tmp_folder_video_input = "./tmp_data/video/input"
tmp_folder_video_output = "./tmp_data/video/output"
os.makedirs(tmp_file_folder_input, exist_ok=True)
os.makedirs(tmp_output_img_folder, exist_ok=True)
os.makedirs(tmp_output_lbl_folder, exist_ok=True)
os.makedirs(tmp_folder_video_input, exist_ok=True)
os.makedirs(tmp_folder_video_output, exist_ok=True)
s3_client = boto3.client('s3')

triton_client = None


@app.get("/", tags=["Health Check"])
async def root():
    return {"inference_service_health": "Ok"}

#The inference-service endpoint receives post requests with the image and returns the transformed image
@app.get("/detect/", tags=["Object Detect Photo"])
async def detect(input_image_file_url: str, output_image_folder_url: str, output_label_folder_url: str):
    #We read the file and decode it
    # s3://aerial-detection-mlops4/inferencing/photos/input/19d09312c52945f8bcdd283c627d9b44-9999942_00000_d_0000214.jpg
    bucket_name, key_name_without_file, file_name = parse_s3_url(unquote(input_image_file_url))
    
    temp_input_image_filename = f'{tmp_file_folder_input}{os.sep}{file_name}'
    temp_output_image_filename = f'{tmp_output_img_folder}{os.sep}OUT-{file_name}'
    temp_output_label_filename = f'{tmp_output_lbl_folder}{os.sep}OUT-{os.path.splitext(file_name)[0]}.txt'
    s3_client.download_file(Bucket = bucket_name, Key = f'{key_name_without_file}/{file_name}', Filename = temp_input_image_filename)
    if logger.isEnabledFor(level=logging.DEBUG):
        logger.debug(f'created local temp file : {temp_input_image_filename}')        
        logger.debug(f'bucket_name = {bucket_name}, key_name_without_file = {key_name_without_file}, file_name = {file_name}')
        logger.debug("input_image_file_url: " + unquote(input_image_file_url))
        logger.debug("output_image_file_url: " + unquote(output_image_folder_url))
        logger.debug("output_label_file_url: " + unquote(output_label_folder_url))

    try:
        start_time = time.time()
        get_triton_client().detect_image(input_image_file=temp_input_image_filename, output_image_file=temp_output_image_filename, output_label_file=temp_output_label_filename)
        logger.info(f"Time taken to run detect_image method: {int((time.time()-start_time)*1000)} milli seconds")
        out_bucket_name, out_key_name_without_file, new_out_image_file_name_only = parse_s3_url(f"{unquote(output_image_folder_url)}/{temp_output_image_filename.split('/')[-1]}")
        s3_client.upload_file(Bucket = out_bucket_name, Filename = temp_output_image_filename, Key = f'{out_key_name_without_file}/{new_out_image_file_name_only}')
        if os.path.exists(temp_output_label_filename):
            out_label_bucket_name, out_label_key_name_without_file, new_out_label_file_name_only = parse_s3_url(f"{unquote(output_label_folder_url)}/{temp_output_label_filename.split('/')[-1]}")
            s3_client.upload_file(Bucket = out_label_bucket_name, Filename = temp_output_label_filename, Key = f'{out_label_key_name_without_file}/{new_out_label_file_name_only}')
    except Exception as e:
        logger.warn("Exception encountered: " + str(e))
    finally:
        delete_temp_files([temp_input_image_filename, temp_output_image_filename, temp_output_label_filename])
    return {"input_image_file_url": input_image_file_url,
             "output_image_file_url": f's3://{out_bucket_name}/{out_key_name_without_file}/{new_out_image_file_name_only}',
             "output_label_file_url": f's3://{out_label_bucket_name}/{out_label_key_name_without_file}/{new_out_label_file_name_only}'
            }

#The inference-service endpoint receives post requests with the image and returns the transformed video
@app.get("/detect_video/", tags=["Object Detect Video"])
async def detect_video(input_video_file_url: str, output_video_folder_url: str):
    #We read the file and decode it    
    bucket_name, key_name_without_file, file_name = parse_s3_url(unquote(input_video_file_url))
    
    temp_input_video_filename = f'{tmp_folder_video_input}{os.sep}{file_name}'
    temp_output_video_filename = f'{tmp_folder_video_output}{os.sep}OUT-{file_name}'
    s3_client.download_file(Bucket = bucket_name, Key = f'{key_name_without_file}/{file_name}', Filename = temp_input_video_filename)
    
    logger.info(f'created local temp file : {temp_input_video_filename}')        
    logger.debug(f'bucket_name = {bucket_name}, key_name_without_file = {key_name_without_file}, file_name = {file_name}')
    logger.debug("input_image_file_url: " + unquote(input_video_file_url))
    logger.debug(f"output_video_folder_url: {unquote(output_video_folder_url)}")

    try:
        start_time = time.time()
        get_triton_client().detect_video(input_video_file=temp_input_video_filename, output_video_file=temp_output_video_filename)
        logger.info(f"Time taken to run detect_video method: {int((time.time()-start_time)*1000)} milli seconds")
        out_bucket_name, out_key_name_without_file, new_out_image_file_name_only = parse_s3_url(f"{unquote(output_video_folder_url)}/{temp_output_video_filename.split('/')[-1]}")
        s3_client.upload_file(Bucket = out_bucket_name, Filename = temp_output_video_filename, Key = f'{out_key_name_without_file}/{new_out_image_file_name_only}')
    except Exception as e:
        logger.warn("Exception encountered: " + str(e))
    finally:
        delete_temp_files([temp_input_video_filename, temp_output_video_filename])
    return {"input_video_file_url": input_video_file_url,
             "output_video_file_url": f's3://{out_bucket_name}/{out_key_name_without_file}/{new_out_image_file_name_only}'
            }

def parse_s3_url(s3_path: str):
    s3_path_split = s3_path.split('/')
    bucket_name = s3_path_split[2]
    key_name_without_file = '/'.join(s3_path_split[3:-1])
    file_name = s3_path_split[-1]
    return bucket_name, key_name_without_file, file_name

def get_triton_client():
    global triton_client
    if triton_client is None:
        triton_client = TritonClient(model = my_model, triton_url = my_triton_url)
    return triton_client

def delete_temp_files(file_name_array):
    for f in file_name_array:
        try:
            if os.path.exists(f):
                os.remove(f)
                logger.debug(f'Deleted temp file: {f}')
        except Exception as e:
            logger.warn("Error deleting temp file: " + str(e))



