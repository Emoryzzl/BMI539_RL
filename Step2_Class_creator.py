# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:41:57 2024
@author: zzha962
"""
import numpy as np
import pandas as pd
import os

eval_index_name = ['PA','IoU','Precision','Recall','DICE'] 
model_name_list = ['unet3d', 'residual_unet3d', 'highresnet', 'densenet3d', 'densevoxelnet3d', 'vnet3d']

Seg_eval_table_path = '../../Comparative_models/Eval_result_for_RL/'
Seg_index = 'DICE'
eval_matrix_table = np.zeros((100,6),dtype=np.float32)   # 100 cases, 6 models
for i in range(len(model_name_list)):
    model_name = model_name_list[i]
    eval_table_df = pd.read_csv(os.path.join(Seg_eval_table_path,'{}_eval.csv'.format(model_name)))    
    eval_matrix_table[:,i] = eval_table_df[Seg_index] 

eval_matrix_rank = np.argsort(np.argsort(eval_matrix_table, axis=1), axis=1) + 1  # +1 to make ranks start from 1

Thresholding_selection = 0.9 # Model selection
class_table = np.zeros((100,6),dtype=np.uint8)   # 100 cases, 6 models
class_table[(eval_matrix_rank==6)|(eval_matrix_rank==5)|(eval_matrix_table>Thresholding_selection)] = 1

class_df = pd.DataFrame(data=class_table,columns=model_name_list)
case_name_list = list(eval_table_df['CaseName'])
class_df.insert(0, 'CaseName', case_name_list)
class_df.to_csv('{}_class.csv'.format(Seg_index), index=False)






