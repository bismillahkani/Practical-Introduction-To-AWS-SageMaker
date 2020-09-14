import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import numpy as np

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torch.utils.data

import logging
import requests

import io
import glob
import time
from base64 import b64decode
from io import BytesIO
import base64

from PIL import Image

import boto3

from model import Cifar10CnnModel, predict_image, classes
from utils import get_default_device, to_device, get_transform

# set the constants for the content types
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    device = get_default_device()
    model = Cifar10CnnModel()

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))

    model.to(device).eval()

    print("Done loading model.")
    return model

def decode_uri(uri_string):
    header, encoded = uri_string.split(",", 1)
    img_bytes = b64decode(encoded)
    return img_bytes

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    # process an image uploaded to the endpoint
    if content_type == JPEG_CONTENT_TYPE:
        img = Image.open(io.BytesIO(request_body))        
        return img
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
#         img_request = requests.get(request_body, stream=True)
        img_bytes = decode_uri(request_body)
        img = Image.open(io.BytesIO(img_bytes))
        return img        
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    if accept == JSON_CONTENT_TYPE:
        output = json.dumps(prediction)
        return output, accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))  

def predict_fn(input_object, model):
    
    print('Inferring class of input data.') 
    
    tfms = get_transform()
    
    input_data = tfms(input_object)
    
    device = get_default_device()
    
    output = predict_image(input_data, model)  
    
    return output

 
