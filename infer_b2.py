from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")


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



url = "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80"

image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits.cpu()

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]
pred_seg = pred_seg.numpy().astype(np.uint8)

print(pred_seg.shape)
print(np.unique(pred_seg))
print(type(pred_seg))

pred_seg = b2_label2mask(np.array(image), pred_seg)
pred_seg = Image.fromarray(pred_seg)
pred_seg.save("./output_images/b2_test_output.png")
