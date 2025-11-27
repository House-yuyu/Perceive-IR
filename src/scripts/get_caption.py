
import os
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


lq_path = "datasets/val/sonwy/Snow100K-S/lq"  
degradation = "snow"  

data = []
cnt = 0
total = len(os.listdir(lq_path))
for root, dirs, files in os.walk(lq_path):
    for file in files:
        cnt += 1
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):

            lq_file_path = os.path.join(root, file)

            try:
                image = Image.open(lq_file_path).convert("RGB")
                inputs = processor(image, return_tensors="pt")
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Error processing {lq_file_path}: {e}")
                caption = "" 
                
            print(f'[{cnt} / {total}] {caption}')

            gt_path = lq_file_path.replace('lq','gt')
           
            print(gt_path)

            assert os.path.exists(lq_file_path)
            assert os.path.exists(gt_path)

            data.append({
                "gt_path": gt_path,  
                "lq_path": lq_file_path,  
                "caption": caption,
                "degradation": degradation  
            })

df = pd.DataFrame(data)
df.to_csv("Snow100K-S_LQ_caption.csv", index=False, sep="\t")

