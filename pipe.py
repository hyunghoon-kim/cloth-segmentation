import requests
import base64
import json
import os
from datetime import datetime
import time
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu
from networks import U2NET
from transformers import SegformerImageProcessor
from transformers import AutoModelForSemanticSegmentation
import torch.nn as nn
import random


n_data = 1024

url = "http://127.0.0.1:8191"
device = "cuda"
json_file_path = './payloads/payload_fasion.json'
u2_checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm.pth")

chunk_dir = "chunk_man"
image_dir = "images_man"
caption_dir = "captions_man"
condition_dir = "conditions_man"

u2net = U2NET(in_ch=3, out_ch=4)
u2net = load_checkpoint_mgpu(u2net, u2_checkpoint_path)
u2net = u2net.to(device)
u2net = u2net.eval()

transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)


processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def bgr2pil(bgr_array:np.ndarray) -> Image:
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_array)
    return pil_img


def pil2bgr(pil_img:Image) -> np.ndarray:
    rgb_array = np.array(pil_img)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def u2_label2mask(org_img, label, debug=True) -> np.ndarray:
    mask = np.zeros_like(org_img)
    if debug:
        mask[label==1] = [255, 0, 0] # whole
        mask[label==2] = [0, 255, 0] # upper
        mask[label==3] = [0, 0, 255] # under
    else:
        mask[label==1] = [255, 255, 255] # whole
        mask[label==2] = [255, 255, 255] # upper
        mask[label==3] = [255, 255, 255] # under
    return mask


def b2_label2mask(org_img, label, debug=True) -> np.ndarray:
    if debug:
        decode_map = {
            0: [0, 0, 0],        # Background
            1: [255, 0, 0],      # Hat
            2: [0, 255, 0],      # Hair
            3: [0, 0, 255],      # Sunglasses
            4: [255, 255, 0],    # Upper-clothes
            5: [255, 0, 255],    # Skirt
            6: [0, 255, 255],    # Pants
            7: [128, 0, 0],      # Dress
            8: [0, 128, 0],      # Belt
            9: [0, 0, 128],      # Left-shoe
            10: [128, 128, 0],   # Right-shoe
            11: [128, 0, 128],   # Face
            12: [0, 128, 128],   # Left-leg
            13: [64, 64, 64],    # Right-leg
            14: [192, 192, 192], # Left-arm
            15: [64, 0, 0],      # Right-arm
            16: [0, 64, 0],      # Bag
            17: [0, 0, 64]       # Scarf
        }
    else:
        decode_map = {
            0: [0, 0, 0],        # Background
            1: [0, 0, 0],      # Hat
            2: [0, 0, 0],      # Hair
            3: [0, 0, 0],      # Sunglasses
            4: [255, 255, 255],    # Upper-clothes
            5: [255, 255, 255],    # Skirt
            6: [255, 255, 255],    # Pants
            7: [255, 255, 255],      # Dress
            8: [255, 255, 255],      # Belt
            9: [0, 0, 0],      # Left-shoe
            10: [0, 0, 0],   # Right-shoe
            11: [0, 0, 0],   # Face
            12: [0, 0, 0],   # Left-leg
            13: [0, 0, 0],    # Right-leg
            14: [0, 0, 0], # Left-arm
            15: [0, 0, 0],      # Right-arm
            16: [0, 0, 0],      # Bag
            17: [0, 0, 0]       # Scarf
        }        
    mask = np.zeros_like(org_img)
    for cls_idx, color in decode_map.items():
        mask[label==cls_idx] = color
    return mask


def ensemble_preds(pred1, pred2) -> np.ndarray:
    '''
    pred1과 같은 제로 캔버스를 만들고
    pred2의 [255,255,255]인 영역을 contours로 만든 영역이 candidation area
    candidation area에서 pred2가 [0,0,0]인 영역은 [0,0,0]
    candidation area에서 pred2가 [255,255,255]인 영역은 [255,255,255]
    candidation area에서 pred1이 [255,255,255]인 영역은 [255,255,255]
    을 만들어 리턴하는 함수
    '''
    # 1. 제로 캔버스 생성
    pred3 = np.zeros_like(pred1)

    # 2. pred1에서 흰색 픽셀 좌표 추출
    white_mask_pred1 = np.all(pred2 == 255, axis=-1).astype(np.uint8) * 255
    
    # 3. findContours를 사용하여 pred1의 흰색 영역 윤곽선 찾기
    contours, hierarchy = cv2.findContours(white_mask_pred1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if contours:
        cv2.drawContours(white_mask_pred1, contours, -1, color=1, thickness=cv2.FILLED)
    
    # 4. candidate_area를 위한 마스크 초기화
    candidate_mask = np.zeros(pred1.shape[:2], dtype=np.uint8)

    # 5. 찾은 윤곽선을 채워 candidate_mask 생성
    if contours:
        cv2.drawContours(candidate_mask, contours, -1, color=1, thickness=cv2.FILLED)

    # Candidate area 내의 인덱스 (Boolean 마스크)
    cand_idx = candidate_mask.astype(bool)

    # 6. Candidate area 내 조건에 따라 흰색 영역 결정
    white_pred2 = np.all(pred2 == 255, axis=-1) & cand_idx
    white_pred1 = np.all(pred1 == 255, axis=-1) & cand_idx

    # 7. 조건에 맞는 픽셀에 흰색 할당
    pred3[white_pred2] = [255, 255, 255]
    pred3[white_pred1] = [255, 255, 255]    

    return pred3


def make_condition_img(img, mask, thr1=100, thr2=200) -> np.ndarray:
    prob = mask/255.
    white_img = np.ones_like(img) * 255
    condition_img = img * prob + white_img * (1-prob)
    condition_img = np.clip(condition_img, 0, 255).astype(np.uint8)
    condition_img = cv2.Canny(condition_img, thr1, thr2)
    return condition_img


if __name__ == "__main__":
    for order in range(n_data):
        ## make dirs
        if not os.path.exists(image_dir):
            os.makedirs(image_dir, exist_ok=True)

        if not os.path.exists(chunk_dir):
            os.makedirs(chunk_dir, exist_ok=True)

        if not os.path.exists(condition_dir):
            os.makedirs(condition_dir, exist_ok=True)
        
        if not os.path.exists(caption_dir):
            os.makedirs(caption_dir, exist_ok=True) 

        ## load json
        with open(json_file_path, 'r', encoding='utf-8') as file:
            payload = json.load(file)

        ## assign prompt
        candidate_prompts = ["a", "b", "c"]

        payload["prompt"] = random.choice(candidate_prompts)

        print(f"{order+1}:", payload["prompt"], flush=True)
        ## gen image
        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
        r = response.json()
        num_images = len(r['images'])

        ## decode
        for idx in range(num_images):
            img_byte = base64.b64decode(r['images'][idx])
            img_array = np.frombuffer(img_byte, dtype=np.uint8) # (hwc, )
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        ## u2net inference
        img_pil = bgr2pil(img_bgr)
        img_tensor = transform_rgb(img_pil)
        img_tensor = torch.unsqueeze(img_tensor, 0)

        output_tensor = u2net(img_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()
        
        pred1 = u2_label2mask(img_bgr, output_arr, debug=False)
        
        ## b2net inference
        inputs = processor(images=img_pil, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
        logits,
        size=img_pil.size[::-1],
        mode="bilinear",
        align_corners=False,)

        output_arr2 = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        pred2 = b2_label2mask(img_bgr, output_arr2, debug=False)

        ## pred ensemble
        pred3 = ensemble_preds(pred1, pred2)


        ## gen condition
        condition_img = make_condition_img(img_bgr, pred3)
        condition_img = np.stack([condition_img, condition_img, condition_img], axis=-1)


        caption = payload["prompt"]
        overlay = cv2.addWeighted(img_bgr, 0.5, pred3, 0.5, 0)
        chunk = np.concatenate([img_bgr, pred3, overlay, condition_img], axis=1)    

        filename = timestamp()

        ## save files
        cv2.imwrite(f"{chunk_dir}/{filename}.png", chunk)
        cv2.imwrite(f"{image_dir}/{filename}.png", img_bgr)
        cv2.imwrite(f"{condition_dir}/{filename}.png", cv2.cvtColor(condition_img, cv2.COLOR_RGB2GRAY))
        with open(f"{caption_dir}/{filename}.txt", "w") as f:
            f.write(caption)
    
