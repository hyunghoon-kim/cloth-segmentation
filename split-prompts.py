import os
import shutil
import random

output_dir = "prompts2"

with open("chunk-prompt.txt", "r") as f:
    prompts = f.read()

prompt_list = prompts.strip().split("\n")
print(len(prompt_list))

if os.path.exists(output_dir) == False:
    os.makedirs(output_dir, exist_ok=True)

for idx, prompt in enumerate(prompt_list):
    filename = str(idx).zfill(4) + ".txt"
    with open(f"{output_dir}/{filename}", "w") as f:
        f.write(prompt)
    # break