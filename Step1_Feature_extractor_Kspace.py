# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 01:35:47 2024
@author: zzha962
"""
import torch
import numpy as np
from Utils import data_preprocess
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", device)

''' Read and pre-process the input data '''
img_path = '../../Comparative_models/datasets/CZTZ/train/mri/'
center_size = 32
case_list = os.listdir(img_path)
case_names = []

feature_data = np.zeros((100,center_size**3*2))  # 16*16*16 *2 (magnitude, phase)
for i in range(len(case_list)):
    case_name = case_list[i]
    case_names.append(case_name[:-4])
    mri_volume = data_preprocess(os.path.join(img_path,case_name))
    mri_volume = mri_volume.squeeze_()
    # Step 1: Compute 3D Fourier Transform
    k_space = torch.fft.fftn(mri_volume, dim=(-2, -1, 0))
    
    # Step 2: Shift the zero frequency component to the center
    k_space_shifted = torch.fft.fftshift(k_space, dim=(-2, -1, 0))
    
    # Step 3: Extract magnitude and phase
    start = [(dim - center_size) // 2 for dim in k_space_shifted.shape]
    end = [start[i] + center_size for i in range(3)]
    
    central_k_space = k_space_shifted[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2]
    ]
    
    central_magnitude = torch.abs(central_k_space).flatten().data.cpu().numpy()
    central_phase = torch.angle(central_k_space).flatten().data.cpu().numpy()

    central_magnitude = np.log1p(central_magnitude)
    
    
    feature_data[i,:center_size**3] = central_magnitude
    feature_data[i,center_size**3:] = central_phase
    print('{}/{} done'.format(i+1, len(case_list)))

df = pd.DataFrame(data=feature_data)
df.insert(0, 'CaseName', case_names)
df.to_csv('Image_features2.csv', index=False)
