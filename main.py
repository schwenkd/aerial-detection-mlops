from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import requests

# Let's generate a new FastAPI app
# Generate a FastAPI instance called `app` with the title 'Triton Health Check'
# https://fastapi.tiangolo.com/
app = FastAPI(title='Root Type Triton Health Check')


print("wheres this happening")
#Call your get function for a health Check
@app.get("/", tags=["Health Check in app.get"])
async def root():
    print("Trying a request to yolov5")
    yolov5_response = requests.get("http://yolov5-service:8000/")
    print("got a request from yolov5")
    return yolov5_response.json()
