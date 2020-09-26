try:
    import unzip_requirements
except ImportError:
    pass
print('here')

import copy
import numpy as np
import re
import os
import io
import boto3
import json
import base64
import onnxruntime
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from requests_toolbelt.multipart import decoder

print('then here')
from PIL import Image, ImageDraw

print('Import END...')

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

datname = r'cars.quantized.onnx'

print("Loading Model")
ort_session = onnxruntime.InferenceSession(datname)
print("Model Loaded...")


def get_prediction():
    z = np.random.randn(36,100).astype(np.float32)

    ort_inputs = {ort_session.get_inputs()[0].name: z}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_out1 = ort_outs[0]
    img = np.transpose(ort_out1,(0,2,3,1))
    
    plt.figure(figsize = (6,6))
    gs1 = gridspec.GridSpec(6,6)
    gs1.update(wspace=0.0, hspace=0.0)
    for i in range(1,17):
        plt.subplot(4,4,i)
        plt.axis('off')
        plt.imshow((img[2*i+3]+1)/2., 'gray')
    
    
    in_mem_file = io.BytesIO()
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    pil_img = Image.open(buf)
    pil_img = pil_img.convert('RGB')
    pil_img.save(in_mem_file, format='jpeg')
    in_mem_file.seek(0)
    s3.Object('gdeotale-session6-cars', 'cars.jpg').put(Body=in_mem_file,ContentType='image/JPG', ACL='public-read')
    url = "https://{}.s3.amazonaws.com/{}".format('gdeotale-session6-cars', 'cars.jpg')

    ##url = 'https://gdeotale-session6-cars.s3.ap-south-1.amazonaws.com/cars.jpg'
    print(url)
    buf.close()
    return url


def get_cars(event, context):
    try:
        prediction = get_prediction()
        print("Prediction done")
        
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type':'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': "Predicted Cars", 'image URL':prediction})
        }

    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type':'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials':True
            },
            "body": json.dumps({"error": repr(e)})
        }



