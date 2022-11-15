FROM python:3.8.13
COPY requirements.txt /
RUN pip3 install -r /requirements.txt
COPY . /aerial-detection-client
WORKDIR /aerial-detection-client
EXPOSE 8080
ENTRYPOINT ["./gunicorn.sh"]
# CMD ["uvicorn", "application:app","--host", "0.0.0.0"]