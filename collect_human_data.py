import requests
import base64
import json
import os
from datetime import datetime
import time
from glob import glob
from tqdm import tqdm

# timestamp
def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

# Define the URL and the payload to send.
# JSON path
url = "http://127.0.0.1:8191"
json_file_path = './payloads/payload_xl_base.json'
dirname = 'img_human_reg'

# if not exist dir, make dir
if os.path.exists(dirname) == False:
    os.makedirs(dirname, exist_ok=True)


# JSON load_json
with open(json_file_path, 'r', encoding='utf-8') as file:
    payload = json.load(file)

# prompts load
with open('prompts/caption_whole.txt', 'r', encoding='utf-8') as f:
    prompt_list = [line.strip() for line in f if line.strip()]


print(f"len(prompt_list): {len(prompt_list)}")

for order, prompt in tqdm(enumerate(prompt_list),
                        total=len(prompt_list),
                        bar_format='{l_bar}{bar} | {n}/{total} ({percentage:3.0f}%) - {rate_fmt}'):
    # 처리 코드
    payload['prompt'] = "photo-realistic, 8k, " + prompt 
    # print(payload['prompt'], flush=True)
    
    # Send said payload to said URL through the API.
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()

    num_images = len(r['images'])
    current_time = timestamp()

    # Decode and save the image.
    for idx in range(num_images):
        with open(f"{dirname}/{current_time}_{str(idx).zfill(4)}.jpg", 'wb') as f:
            f.write(base64.b64decode(r['images'][idx]))
    
    # Save prompt as text file
    for idx in range(num_images):
        with open(f"{dirname}/{current_time}_{str(idx).zfill(4)}.txt", 'w') as f:
            f.write(prompt)
    
    # if order == 3:
        # break