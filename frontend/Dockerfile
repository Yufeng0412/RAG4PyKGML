FROM python:3.11-slim

RUN apt-get -y update && \
    apt-get -y install python3-pip

# pip install the other python requirements in the container
WORKDIR .
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install nbformat gradio==5.34.0

# copy the src code into the directory in the container
COPY . /app/
WORKDIR /app

ENV PYTHONUNBUFFERED 1

ENTRYPOINT gunicorn --worker-class uvicorn.workers.UvicornWorker \
  frontend_server:app \
  --workers 1 \
  --bind 0.0.0.0:${APP_BIND_PORT} \
  --log-level info