try:
    import unzip_requirements
except ImportError:
    pass
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN
import boto3
import os
import io
import json
import base64
import numpy as np
from requests_toolbelt.multipart import decoder
print('Import END...')

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'gdeotale-session4-facenet'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'Modeljit.pt'

print('Downloading model...')

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) !=True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        model = torch.jit.load(bytestream)
        print("Model Loaded...")
except Exception as e:
    print(repr(e))
    raise(e)

def get_input_image(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    
    print('Content-Type', content_type_header)
    print('Loading body...')
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final picture that will be used by the model
    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    print('Picture obtained')
    
    return picture

def transform_image(image):
    try:
        transformations = transforms.Compose([
            transforms.Resize(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        image = Image.open(io.BytesIO(image))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

json_file_path = r'imagenet-simple-labels.json'
imagenet_labels = []

with open(json_file_path, "r") as read_file:
    imagenet_labels = json.load(read_file)

def get_prediction(image_bytes):
    print("Starting face extraction")
    #face_img = face_extract(image_bytes)
    #print("face extracted")
    tensor = transform_image(image_bytes)
    pred = model(tensor).argmax().item()
    return pred

def classify_image(event, context):
    try:
        picture = get_input_image(event)
        
        prediction = get_prediction(image_bytes=picture.content)
        print("Predicted class id: {0}".format(prediction))

        if len(imagenet_labels) > 0:
            print("Predicted class name: {0}".format(imagenet_labels[prediction]))      
 
        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename)<4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type':'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': filename.replace('"', ''), 'predicted class id':prediction, 'predicted class name':imagenet_labels[prediction]})
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

