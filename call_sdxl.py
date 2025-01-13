import requests
import base64
import json
import os
from datetime import datetime
import time

# timestamp
def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

# Define the URL and the payload to send.
# JSON path
url = "http://127.0.0.1:8191"
json_file_path = './payloads/payload_fasion.json'
dirname = 'foo'


# JSON load_json
with open(json_file_path, 'r', encoding='utf-8') as file:
    payload = json.load(file)

n_iter = payload['n_iter']
batch_size = payload['batch_size']

# Send said payload to said URL through the API.
response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
r = response.json()

# print(r)
num_images = len(r['images'])

# if not exist dir, make dir
if not os.path.exists(dirname):
    os.makedirs(dirname, exist_ok=True)

# Decode and save the image.
for idx in range(num_images):
    with open(f"{dirname}/{timestamp()}_{str(idx).zfill(3)}.jpg", 'wb') as f:
        f.write(base64.b64decode(r['images'][idx]))