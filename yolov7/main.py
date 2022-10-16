from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image
from detect_fast_api import detect
import cv2
import numpy as np
import io

app = FastAPI()

@app.post("/infer-and-return/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    input_file_save_location = f"inference/input/{uploaded_file.filename}"
    output_file_save_location = input_file_save_location.replace('/input/', '/output/')
    with open(input_file_save_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    print({"info": f"file '{uploaded_file.filename}' saved at '{input_file_save_location}'"})
    
    detect(input_file_save_location, output_file_save_location)

    img1 = cv2.imread(input_file_save_location)
    img2 = cv2.imread(output_file_save_location)
    output_to_compare = Image.fromarray(cv2.hconcat([img1, img2]))
    
    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        output_to_compare.save(buf, format='PNG')
        im_bytes = buf.getvalue()
        
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(im_bytes, headers=headers, media_type='image/png')