FROM python:3.8.13
COPY requirements.txt  .
RUN  pip3 install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app","--host", "0.0.0.0"]