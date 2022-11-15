
# from .prediction import get_prediction, create_output_image
from flask import Flask, request, Response, render_template, flash, request,redirect, url_for, send_from_directory
from flask import current_app as app
import mimetypes
import uuid
import boto3
import os
import logging
import requests
import json
from urllib.parse import unquote
import cv2
import re

MB = 1 << 20
BUFF_SIZE = 10 * MB
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_PHOTO_FILE_EXTENSIONS = {'jpg'}
ALLOWED_VIDEO_FILE_EXTENSIONS = {'mp4'}
logger = logging.getLogger(__name__)
PHOTO_INFERENCE_SERVICE_ENDPOINT = "http://inference-service:8000/detect"
VIDEO_INFERENCE_SERVICE_ENDPOINT = "http://inference-service:8000/detect_video"

S3_BUCKET = "aerial-detection-mlops4"
PHOTO_INPUT_S3_KEY =  "inferencing/photos/input"
PHOTO_OUTPUT_S3_IMAGES_KEY =  "inferencing/photos/output/images"
PHOTO_OUTPUT_S3_LABELS_KEY =  "inferencing/photos/output/labels"

VIDEO_INPUT_S3_KEY =  "inferencing/videos/input"
VIDEO_OUTPUT_S3_IMAGES_KEY =  "inferencing/videos/output"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
tmp_file_folder_name = "/static/tmp_data"
# tmp_file_folder_name = "/static"
tmp_file_folder = f'{ROOT_DIR}{tmp_file_folder_name}'
os.makedirs(tmp_file_folder, exist_ok=True) 

s3_client = boto3.client('s3')

@app.route('/', methods=['GET', 'POST'])
def aerial_ai():
    # Write the GET Method to get the index file
    if request.method == 'GET':
        return render_template('index.html')
    # Write the POST Method to post the results file
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('File Not Uploaded')
            return
        # Read file from upload
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_photo_file(file.filename):
            local_output_file_name = handle_detect_photo(file)
            return render_template('result.html', input_file_name=file.filename, output_file_name = local_output_file_name, show_photo=True)
        elif file and allowed_video_file(file.filename):
            local_video_output_file_name = handle_detect_video(file)
            return render_template('result.html', input_file_name=file.filename, output_file_name = local_video_output_file_name, show_photo=False)
            
        return redirect(request.url)


def handle_detect_photo(file):
    # Assign an id to the asynchronous task
    task_id = uuid.uuid4().hex
    new_file_name = f'{task_id}-{file.filename}'
    img = request.files['file']
    if img:
        try:
            # os.makedirs(tmp_file_folder, exist_ok=True) 
            local_input_file_path = os.path.join(tmp_file_folder, new_file_name)
            img.save(local_input_file_path)
            s3_client.upload_file(Bucket = S3_BUCKET, Filename = local_input_file_path, Key = f'{PHOTO_INPUT_S3_KEY}/{new_file_name}')
            data = {"input_image_file_url": f's3://{S3_BUCKET}/{PHOTO_INPUT_S3_KEY}/{new_file_name}',
                    "output_image_folder_url": f's3://{S3_BUCKET}/{PHOTO_OUTPUT_S3_IMAGES_KEY}',
                    "output_label_folder_url": f's3://{S3_BUCKET}/{PHOTO_OUTPUT_S3_LABELS_KEY}'
                    }
            
            response = requests.get(url = PHOTO_INFERENCE_SERVICE_ENDPOINT, params = data)
            # os.remove(local_file_name)
            logger.info(f"Successfully handled {new_file_name}")
            # logger.info(f'URL for static file:{url_for("static", filename ="images/prediction.png")}')
            dict = json.loads(response.text)
            output_image_file_url = dict["output_image_file_url"]
            output_label_file_url = dict["output_label_file_url"]
            logger.info(f's3-output image-file is : {output_image_file_url}')
            logger.info(f's3-output label-file is : {output_label_file_url}')
            bucket_name, key_name_without_file, output_file_name = parse_s3_url(unquote(output_image_file_url))
            # local_output_file_name = f'{tmp_file_folder}{os.sep}{file_name}'
            
            
            local_output_file_path = os.path.join(tmp_file_folder, output_file_name)

            logger.info(f'Local output-file path is: {local_output_file_path}')
            s3_client.download_file(Bucket = bucket_name, Key = f'{key_name_without_file}/{output_file_name}', Filename = local_output_file_path)
            combined_out_file_path = create_combined_image(local_input_file_path, local_output_file_path)
            logger.info(f'Combined output-file path is: {local_output_file_path}')
            local_output_file_name = f"{tmp_file_folder_name}/{combined_out_file_path.split('/')[-1]}"
            # if os.path.exists(local_output_file_path):
            #     os.remove(local_output_file_path)
        except requests.exceptions.HTTPError as errh:
            logger.info("Http Error:",errh)
        except requests.exceptions.ConnectionError as errc:
            logger.info("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            logger.info("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            logger.info("OOps: Something Else",err)
        # except OSError as e:
        #     logger.warn ("Error deleting file: %s - %s." % (e.filename, e.strerror))
        except Exception as e:
            logger.warn(e)

    # return the local_output_file_name
    return local_output_file_name


def handle_detect_video(file):
    # Assign an id to the asynchronous task
    task_id = uuid.uuid4().hex
    new_video_file_name = f'{task_id}-{file.filename}'
    img = request.files['file']
    if img:
        try:
            # os.makedirs(tmp_file_folder, exist_ok=True) 
            local_input_file_path = os.path.join(tmp_file_folder, new_video_file_name)
            img.save(local_input_file_path)
            s3_client.upload_file(Bucket = S3_BUCKET, Filename = local_input_file_path, Key = f'{VIDEO_INPUT_S3_KEY}/{new_video_file_name}')
            data = {"input_video_file_url": f's3://{S3_BUCKET}/{VIDEO_INPUT_S3_KEY}/{new_video_file_name}',
                    "output_video_folder_url": f's3://{S3_BUCKET}/{VIDEO_OUTPUT_S3_IMAGES_KEY}'
                    }
            
            response = requests.get(url = VIDEO_INFERENCE_SERVICE_ENDPOINT, params = data)
            # os.remove(local_file_name)
            logger.info(f"Successfully handled {new_video_file_name}")
            # logger.info(f'URL for static file:{url_for("static", filename ="images/prediction.png")}')
            dict = json.loads(response.text)
            output_video_file_url = dict["output_video_file_url"]
            logger.info(f's3-output image-file is : {output_video_file_url}')
            bucket_name, key_name_without_file, output_file_name = parse_s3_url(unquote(output_video_file_url))
          
            local_output_file_path = os.path.join(tmp_file_folder, output_file_name)
            logger.info(f'Local output-file path is: {local_output_file_path}')
            s3_client.download_file(Bucket = bucket_name, Key = f'{key_name_without_file}/{output_file_name}', Filename = local_output_file_path)
            # local_output_file_name = f"{tmp_file_folder_name}/{local_output_file_path.split('/')[-1]}"
            local_output_file_name = local_output_file_path.split('/')[-1]
        except requests.exceptions.HTTPError as errh:
            logger.info("Http Error:",errh)
        except requests.exceptions.ConnectionError as errc:
            logger.info("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            logger.info("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            logger.info("OOps: Something Else",err)
        # except OSError as e:
        #     logger.warn ("Error deleting file: %s - %s." % (e.filename, e.strerror))
        except Exception as e:
            logger.warn(e)

    # return the local_output_file_name
    return local_output_file_name


def parse_s3_url(s3_path: str):
    s3_path_split = s3_path.split('/')
    bucket_name = s3_path_split[2]
    key_name_without_file = '/'.join(s3_path_split[3:-1])
    file_name = s3_path_split[-1]
    return bucket_name, key_name_without_file, file_name

def create_combined_image(input_file_save_location, output_file_save_location):
    img1 = cv2.imread(input_file_save_location)
    img2 = cv2.imread(output_file_save_location)
    h_img = cv2.hconcat([img1, img2])
    combined_out_file_location = output_file_save_location.replace('/OUT-', '/COMBINED-OUT-')
    cv2.imwrite(combined_out_file_location, h_img)
    return combined_out_file_location


def allowed_photo_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_PHOTO_FILE_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_FILE_EXTENSIONS

@app.route("/display/<path:filename>")
def display_video(filename):
    video_path = os.path.join(tmp_file_folder, filename)
    return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/replay/<path:filename>/<path:input_filename>")
def replay_video(filename, input_filename):
    return render_template('result.html', input_file_name=input_filename, output_file_name = filename, show_photo=False)

# https://stackoverflow.com/questions/72367168/is-it-possible-to-use-opencv-to-display-a-video-on-web-browser-while-blurring-th
def gen_frames(video_file_path):  # generate frame by frame from camera
    camera = cv2.VideoCapture(video_file_path) 
    print(f'gen_frames from {video_file_path}')
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

