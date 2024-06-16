"""
Step 4
"""

# https://www.machinecurve.com/index.php/2020/11/10/working-with-imbalanced-datasets-with-tensorflow-and-keras/
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

import pandas as pd
import numpy as np

import os
import sys
from os import listdir
from os.path import isfile, join
import pickle
import json

from keras_preprocessing.sequence import pad_sequences

from random import random

import utility.parameters as param

import version_bert.word2vec_version.utility.parameters as param2

path_to_chunks_ints = param2.save_delivery_here + "c2_test\\chunk_"+str(param2.chunk_size)+"\\"

chunk_size = param2.chunk_size
path_to_dir_dict = param2.save_delivery_here + "d_test\\chunk_"+str(param2.chunk_size)+"\\chunks_by_book\\"
#with open(path_to_dir_dict + "map_chunks_per_year.json", 'r') as fp:
#        map_chunks_per_year = json.load(fp)
isExist = os.path.exists(path_to_dir_dict)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path_to_dir_dict)
  print("The new directory is created!")

threshold_year = 1800

file_names = [f for f in listdir(path_to_chunks_ints) if isfile(join(path_to_chunks_ints, f))]

map_chunks_by_year = {}

df_with_target = pd.DataFrame(columns=['chunks_ints', 'attention_masks', 'year'])
count = 0

for ind,file_name in enumerate(file_names):
    
    index_of_underscore = file_name.find('_')
    str_target_year = file_name[:index_of_underscore]
    
    int_target_year = int(str_target_year)
    
    count+=1
    try:
        df_temp = pd.read_pickle(os.path.join(path_to_chunks_ints, file_name))
        df_temp['year'] = [int_target_year] * len(df_temp['chunks_ints'])
        if int_target_year not in map_chunks_by_year.keys():
            map_chunks_by_year[int_target_year] = 0
        map_chunks_by_year[int_target_year]+=len(df_temp['chunks_ints'])
        
        df_temp.to_pickle(path_to_dir_dict + str(ind) + '_model_number_books_5_16_train_X_test.p')
        print(str(len(df_temp)) + "   " + str(count) + " year: " + str_target_year +" filename: " + file_name)
    

    except:
        print("Unexpected error:", sys.exc_info()[0])
        continue







