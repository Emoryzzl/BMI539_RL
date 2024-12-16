# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:24:10 2024
@author: zzha962
"""
import medim
import torch
import torch.nn.functional as F
import numpy as np
from Utils import data_preprocess
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", device)

#ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
model = medim.create_model("SAM-Med3D",
                           pretrained=True,
                           checkpoint_path="./sam_med3d_turbo.pth")
model = model.to(device)
model.eval()

''' Read and pre-process the input data '''
img_path = '../../Comparative_models/datasets/CZTZ/train/mri/'
output_dir = "./pred/"
case_list = os.listdir(img_path)
case_names = []

feature_dim = 384*2*2*2*2
#feature_data = np.zeros((100,6144))  # [384, 8, 8, 8] -> [384, 2, 2, 2] * 2 (Avg, Max) -> (384*2*2*2)*2
feature_data = np.zeros((100,feature_dim))  # [384, 8, 8, 8] -> [384, 4, 4, 4] 
for i in range(len(case_list)):
    case_name = case_list[i]
    case_names.append(case_name[:-4])
    roi_image = data_preprocess(os.path.join(img_path,case_name))
    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)

    #Max_embeddings = F.max_pool3d(image_embeddings, kernel_size=2, stride=2)
    #feature_data[i,:] = Max_embeddings.flatten().data.cpu().numpy() 
        
    Avg_embeddings = F.avg_pool3d(image_embeddings, kernel_size=4, stride=4)
    Max_embeddings = F.max_pool3d(image_embeddings, kernel_size=4, stride=4)   
    feature_data[i,:int(feature_dim/2)] = Avg_embeddings.flatten().data.cpu().numpy()
    feature_data[i,int(feature_dim/2):] = Max_embeddings.flatten().data.cpu().numpy()    
    print('{}/{} done'.format(i+1, len(case_list)))

df = pd.DataFrame(data=feature_data)
df.insert(0, 'CaseName', case_names)
df.to_csv('Image_features.csv', index=False)









