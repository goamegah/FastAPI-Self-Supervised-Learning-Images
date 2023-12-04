# SsIma API
SsIma API which aim to implement API in order to serve this [model](). After we
containerize api using docker.

## Table des Matières

1. [Introduction](#introduction)
2. [Project workflows](#project-workflows)
3. [Run model](#run-api)

---

## Introduction
Project concerns of implementing simple deep learning architecture
for classifying ```MNIST``` image data with two constraints
- Train model with only 100 labels samples:
- Implement API to serve model functionality
- Implement Web App (as client) to display model performance.

**API** will be used to serve model by exposing two functionalities:
- Receiving request that will contains image from a **Frontend app** 
and response by **prediction ** which will be digit into the image.
- Getting request that will contain ```model name``` and will response
by posting ***model performance*** 

**This part of project concerns only API implementation and deployment as service
using Docker**.

## Project workflows:

![Alt Text](documentation/local_arch.png)



## Run API
Please before going on, make sure having installed ```uvicorn```, ```python-multipart``` and ```torch```.
If not, you can require librairy by run ```pip install -r requirements.txt```


In the root of project can launch model API by using following command

```shell
$  uvicorn api:app --reload
```

## Containerize app with Docker
for containerize APi app, **dockerfile** is available for building and deploy image

```shell
$ docker build -t image_name:version .
$ docker run -p 8000:8000 image_name:version
```
