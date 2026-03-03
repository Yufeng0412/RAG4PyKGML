## Relevant for supplying routes
from fastapi import FastAPI

## Relevant for supporting docker utilities
import subprocess
import docker

## Relevant for taking in NVAPI_Key
import os
from pydantic import BaseModel, Field, validator
from typing import Any, Dict

client = docker.from_env()
app = FastAPI()

@app.get("/")
async def read_root():
    """Default Route: Usually good to render instructions..."""
    return {"Hello": "World"}


@app.get("/help")
async def read_root():
    """Typical help route to tell users what they can do"""
    return {"Options": "[/containers, /containers/{container_id}/logs, containers/{container_id}/restart]"}


@app.get("/healthy")
async def read_root():
    """Typical health check route. Can be used by other microservices to tell whether they can rely on this one"""
    return True


@app.get("/containers")
async def list_containers():
    """A listing route. Lists current set of containers"""
    containers = client.containers.list(all=True)
    return [{"id": container.id, "name": container.name, "status": container.status} for container in containers]


@app.get("/containers/{container_name}/logs")
async def get_container_logs(container_name: str):
    """Route that allows you to query the log file of the container"""
    try:
        container = client.containers.get(container_name)
        logs = container.logs()
        return {"logs": logs.decode('utf-8')}
    except NotFound:
        return {"error": f"Container `{container_name}` not found"}


@app.post("/containers/{container_name}/restart")
async def restart_container(container_name: str):
    """Just in case it ever becomes necessary (probably won't be though)"""
    try:
        container = client.containers.get(container_name)
        container.restart()
        return {"status": f"Container {container_name} restarted successfully"}
    except NotFound:
        return {"error": f"Container {container_name} not found"}

######################################################################################
## More info: https://fastapi.tiangolo.com/tutorial/body/#import-pydantics-basemodel
class Key(BaseModel):

    ## Possible variables that your message body expects
    nvapi_key: str

    # Validator using custom function
    @validator('nvapi_key')
    def check_nvapi_prefix_function(cls, v):
        if not v.startswith('nvapi-'):
            raise ValueError('nvapi_key must start with "nvapi-"')
        return v

API_KEY = None

@app.post("/set_key/")
async def set_key(key: Key):
    global API_KEY 
    API_KEY = key.nvapi_key 
    return {"result" : "Key set successfully"}

@app.get("/get_key/")
async def get_key():
    global API_KEY
    return {"nvapi_key": API_KEY}