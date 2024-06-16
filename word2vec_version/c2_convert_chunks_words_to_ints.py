"""
This file run the code that response 
about creating vectors of the chunks from the words files
Step 2
"""

import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pickle
from transformers import BertModel, BertForMaskedLM
print("start *****************************************************")
from version_bert.Tokenizer.dictatokenizer import DictaAutoTokenizer
print("end *****************************************************")
import torch

import json
import version_bert.word2vec_version.utility.variable as var
import version_bert.tokenize_version.utility.parameters as param
import version_bert.word2vec_version.utility.parameters as param2
from version_bert.word2vec_version.utility.functions import print_structure_type

model_path = 'C:/Users/User/source/repos/PythonApplication1/PythonApplication1/version_bert/BerelRun1_72580/'
tokenizer = DictaAutoTokenizer.from_pretrained(model_path)



chunk_size = param2.chunk_size
path_to_load_chunks_words = param.save_delivery_here + "c"+param2.version+"\\chunk_"+str(param2.chunk_size)+"\\"
path_to_save_chunks_ints = param2.save_delivery_here + "c2"+param2.version+"\\chunk_"+str(param2.chunk_size)+"\\"
isExist = os.path.exists(path_to_save_chunks_ints)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path_to_save_chunks_ints)
  print("The new directory is created!")

#if param.is_test_set:
#    path_to_load_chunks_words = param.data_dir+'b_chunks_words_'+str(chunk_size)+'_test\\'
#    path_to_save_chunks_ints = param.data_dir+'c_chunks_ints_'+str(chunk_size)+'_test\\'


def applay_list(w):
    return list(w)

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


file_names = [f for f in listdir(path_to_load_chunks_words) if isfile(join(path_to_load_chunks_words, f))]
for file_name in file_names:
    df2 = pd.read_pickle(os.path.join(path_to_load_chunks_words, file_name))
    print(type(df2), " : ", len(df2))
    #print("1. ***********************************")
    #print_structure_type(df2['line'])
    df2['line2'] = df2['line'].apply(tokenizer.encode)
    df2['line2'] = df2['line2'].apply(torch.tensor)
    df2['line2'] = df2['line2'].unsqueeze(0)

    """
    In count:  0  type:  <class 'pandas.core.series.Series'>
        Series_to_list( count:  0  type:  <class 'list'>  len:  184891
    In count:  1  type:  <class 'int'>
        int count:  1  value:  29978
    """
    """
    In count:  0  type:  <class 'pandas.core.series.Series'>
        Series_to_list( count:  0  type:  <class 'list'>  len:  68
    In count:  1  type:  <class 'list'>
        list count:  1  type:  <class 'list'>  len:  149
    In count:  2  type:  <class 'int'>
        int count:  2  value:  1
    """
    
    #print("2. ***********************************")
    #print_structure_type(df2['line2'])
    flat_list = [item for sublist in df2['line2'] for item in sublist]
    #print("3. ***********************************")
    x = flat_list
    ###x = ' '.join(df2['line2'].values).split(' ')
    x = list(divide_chunks(x, chunk_size))
    del x[-1]
    #print_structure_type(x)
    
    df_chunks = pd.DataFrame({'chunks_ints': x})
    df_chunks['chunks_ints'] = df_chunks['chunks_ints'].apply(applay_list)
    df_chunks['len_sent_ints'] = df_chunks['chunks_ints'].apply(len)

    #print("*****************************************************")
    #print("df_chunks['chunks_ints'].shape: " , df_chunks['chunks_ints'].shape)
    #if df_chunks['chunks_ints'].shape[0] > 0:
    #    print_structure_type(df_chunks['chunks_ints'])
    #print("*****************************************************")

    df_chunks.to_pickle(path_to_save_chunks_ints + file_name)
    print(file_name)

