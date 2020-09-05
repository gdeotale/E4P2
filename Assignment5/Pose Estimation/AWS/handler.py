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
from requests_toolbelt.multipart import decoder

print('then here')
from PIL import Image, ImageDraw

print('Import END...')


# S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'session5-body-pose'
# MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'simple_pose_estimation_quantized.onnx'

# print(S3_BUCKET, MODEL_PATH)
# print("Downloading model...")

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

datname = r'simple_pose_estimation_quantized.onnx'
# ort_session = None
# print(os.path.isfile(MODEL_PATH))

# try:
#     if os.path.isfile(MODEL_PATH) == True:
#         obj = s3_client.get_object(Bucket=S3_BUCKET,Key=MODEL_PATH)
#         print("Creating Bytestream")
#         bytestream = io.BytesIO(obj['Body'].read())
#         print("Loading Model")
#         ort_session = onnxruntime.InferenceSession(bytestream)
#         print(ort_session)
#         # model = torch.jit.load(bytestream)
#         print("Model Loaded...")
# except Exception as e:
#     print(repr(e))
#     raise(e)

print("Loading Model")
ort_session = onnxruntime.InferenceSession(datname)
print("Model Loaded...")

def transform_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        np_image = copy.deepcopy(image)
        np_image = np_image.resize((256,256))
        np_image = np.asarray(np_image)
        np_image = np.expand_dims(np_image, axis = 0)
        np_image = np_image.transpose(0, 3, 1, 2)
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(np_image.shape).astype('float32')
        for i in range(np_image.shape[0]):
            for j in range(np_image.shape[1]):
                norm_img_data[i,j,:,:] = (np_image[i,j,:,:]/255 - mean_vec[j]) / stddev_vec[j]

        return norm_img_data

    except Exception as e:
        print( repr(e))
        raise(e)

def get_prediction(image_bytes):
    np_img = transform_image(image_bytes=image_bytes)
    ort_inputs = {ort_session.get_inputs()[0].name: np_img}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = ort_outs[0][0]
    JOINTS = ['0 - r ankle', '1 - r knee', '2 - r hip', '3 - l hip', '4 - l knee', '5 - l ankle', '6 - pelvis', '7 - thorax', '8 - upper neck', '9 - head top', '10 - r wrist', '11 - r elbow', '12 - r shoulder', '13 - l shoulder', '14 - l elbow', '15 - l wrist']
    JOINTS = [re.sub(r'[0-9]+|-', '', joint).strip().replace(' ', '-') for joint in JOINTS]

    get_keypoints = lambda pose_heatmaps : [np.unravel_index(np.argmax(a, axis=None), a.shape) for a in pose_heatmaps]
    POSE_PAIRS = [
    # UPPER BODY
                [9, 8],
                [8, 7],
                [7, 6],

    # LOWER BODY
                [6, 2],
                [2, 1],
                [1, 0],

                [6, 3],
                [3, 4],
                [4, 5],

    # RIGHT ARM
                [7, 12],
                [12, 11],
                [11, 10],

    # LEFT ARM
                [7, 13],
                [13, 14],
                [14, 15]
    ]

    OUT_SHAPE = (64, 64)
    image_original = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(image_original)
    key_points = list(get_keypoints(ort_outs))
    is_joint_plotted = [False for i in range(len(JOINTS))]
    for pose_pair in POSE_PAIRS:
        from_body_part, to_body_part = pose_pair

        ## coordinates of body parts in model output channel size ##
        from_y_body_part, from_x_body_part = key_points[from_body_part]
        to_y_body_part, to_x_body_part = key_points[to_body_part]

        ACTUAL_IMG_HEIGHT, ACTUAL_IMG_WIDTH = image_original.size

        ## coordinates of body parts in actual image channel size ##
        from_x_body_part, to_x_body_part = from_x_body_part * ACTUAL_IMG_HEIGHT / OUT_SHAPE[0], to_x_body_part * ACTUAL_IMG_HEIGHT / OUT_SHAPE[0]
        from_y_body_part, to_y_body_part = from_y_body_part * ACTUAL_IMG_WIDTH / OUT_SHAPE[1], to_y_body_part * ACTUAL_IMG_WIDTH / OUT_SHAPE[1]

        from_x_body_part, to_x_body_part = int(from_x_body_part), int(to_x_body_part)
        from_y_body_part, to_y_body_part = int(from_y_body_part), int(to_y_body_part)

        if not is_joint_plotted[from_body_part]:
            # this is a joint
            draw.ellipse((from_x_body_part-8, from_y_body_part-8, from_x_body_part+8, from_y_body_part+8), fill=(255,0,0,0))
            is_joint_plotted[from_body_part] = True

        if not is_joint_plotted[to_body_part]:
            # this is a joint
            draw.ellipse((to_x_body_part-8, to_y_body_part-8, to_x_body_part+8, to_y_body_part+8), fill=(255,0,0,0))
            is_joint_plotted[to_body_part] = True

        # this is a joint connection, plot a line
        draw.line(((from_x_body_part, from_y_body_part), (to_x_body_part, to_y_body_part)), (0, 0, 255), 4)

    in_mem_file = io.BytesIO()
    image_original.save(in_mem_file, format=image_original.format)
    in_mem_file.seek(0)
    s3.Object('session5-body-pose', 'body-pose.jpg').put(Body=in_mem_file,ContentType='image/JPG', ACL='public-read')
    url = "https://{}.s3.amazonaws.com/{}".format('session5-body-pose', 'body-pose.jpg')
    print(url)
    return url


def estimate_pose(event, context):
    try:
        content_type_header = event['headers']['content-type']
        #print(event['body'])
        body = base64.b64decode(event["body"])
        print('BODY LOADED')
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        print("Picture loaded")	
        prediction = get_prediction(image_bytes=picture.content)
        print("Prediction done")
        print(prediction)
        
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
            "body": json.dumps({'file': filename.replace('"', ''), 'image URL':prediction})
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


