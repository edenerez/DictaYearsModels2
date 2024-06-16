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
path_to_dir_dict = param2.save_delivery_here + "d_test\\chunk_"+str(param2.chunk_size)+"\\all_chunks\\"
#with open(path_to_dir_dict + "map_chunks_per_year.json", 'r') as fp:
#        map_chunks_per_year = json.load(fp)
isExist = os.path.exists(path_to_dir_dict)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path_to_dir_dict)
  print("The new directory is created!")

file_names = [f for f in listdir(path_to_chunks_ints) if isfile(join(path_to_chunks_ints, f))]

map_chunks_by_year = {}

#df_with_target = pd.DataFrame(columns=['len_sent_ints', 'chunks_ints', 'target', 'year'])
df_with_target = pd.DataFrame(columns=['chunks_ints', 'attention_masks', 'year'])
count = 0
for file_name in file_names:
    
    index_of_underscore = file_name.find('_')
    str_target_year = file_name[:index_of_underscore]
    
    int_target_year = int(str_target_year)
    #if int_target_year >1000:
    #    break
    
    count+=1
    print(str(len(df_with_target)) + "   " + str(count) + " year: " + str_target_year +" filename: " + file_name)
    try:
        df_temp = pd.read_pickle(os.path.join(path_to_chunks_ints, file_name))
        
        #print(df_temp.head())
        
        df_temp['year'] = [int_target_year] * len(df_temp['chunks_ints'])
        #df_temp['target'] = [target] * len(df_temp['chunks_ints'])
        df_with_target = pd.concat([df_temp, df_with_target[[ 'chunks_ints', 'attention_masks', 'year']]])

        if int_target_year not in map_chunks_by_year.keys():
            map_chunks_by_year[int_target_year] = 0
        map_chunks_by_year[int_target_year]+=len(df_temp['chunks_ints'])
        #print('{} was loaded'.format(file_name))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        continue

with open(os.path.join(path_to_dir_dict, 'map_chunks_by_year.p'), 'wb') as fp:
    pickle.dump(map_chunks_by_year, fp, protocol=pickle.HIGHEST_PROTOCOL)

df_with_target.to_pickle(path_to_dir_dict + 'model_number_books_5_16_train_X_test.p')

"""   
X_test = df_with_target['chunks_ints']
year_test = df_with_target['year']
with open(os.path.join(path_to_dir_dict, 'model_number_books_5_16_train_X_test_same_files.p'), 'wb') as fp:
    pickle.dump(X_test, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(path_to_dir_dict, 'model_number_books_5_16_train_y_test_year_same_files.p'), 'wb') as fp:
    pickle.dump(year_test, fp, protocol=pickle.HIGHEST_PROTOCOL)
"""

