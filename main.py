# from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import requests

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='FastAPI Main')


#Call your get function for a health Check
#to receive both (face-bokeh and face-emotion)
#The root path will be used as the health check endpoint
@app.get("/", tags=["Health Check"])
async def root():
    response_inference = requests.get("http://inference-service:8000/")
    # response_face_emotion = requests.get("http://webapp:8000/")
    # return [response_face_bokeh.content, response_face_emotion.content]
    return response_inference.content
    # return {"main-health-check":"OK"}