FROM docker:20.10

RUN apk add --no-cache python3 py3-pip

WORKDIR /app
# https://github.com/docker/docker-py/issues/3256
RUN pip install fastapi==0.111.0 uvicorn==0.29.0 docker==7.0.0 pydantic==2.7.1 requests==2.31.0

COPY docker_router.py .

CMD ["uvicorn", "docker_router:app", "--host", "0.0.0.0", "--port", "8070"]
