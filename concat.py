import cv2
import numpy as np
from glob import glob



input_paths = sorted(glob("input_images/*.png"))
output_paths = sorted(glob("output_images/*.png"))



for in_path, out_path in zip(input_paths, output_paths):
    print(in_path.split('/')[1] == out_path.split('/')[1])
    filename = in_path.split('/')[1]
    img_input = cv2.imread(in_path)
    img_output = cv2.imread(out_path)
    img_overlay = cv2.addWeighted(img_input, 0.5, img_output, 0.5, 0)

    img_concat = np.concatenate((img_input, img_output, img_overlay), axis=1)

    cv2.imwrite(f"compare_u2/{filename}", img_concat)