

from version_bert.Tokenizer.dictatokenizer import DictaAutoTokenizer

import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pickle
import version_bert.word2vec_version.utility.variable as var
import version_bert.tokenize_version.utility.parameters as param
import version_bert.word2vec_version.utility.parameters as param2
from version_bert.word2vec_version.utility.functions import print_structure_type

path_to_save_chunks_ints = param2.save_delivery_here + "c2"+param2.version+"\\chunk_"+str(param2.chunk_size)+"\\"
isExist = os.path.exists(path_to_save_chunks_ints)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path_to_save_chunks_ints)
  print("The new directory is created!")

def applay_list(w):
    return list(w)

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

path_to_load_chunks_words = param.save_delivery_here + "c"+param2.version+"\\chunk_"+str(param2.chunk_size)+"\\"

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
model_path = 'C:/Users/User/source/repos/PythonApplication1/PythonApplication1/version_bert/BerelRun1_72580/'
tokenizer = DictaAutoTokenizer.from_pretrained(model_path)

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []
chunk_size = param2.chunk_size
file_names = [f for f in listdir(path_to_load_chunks_words) if isfile(join(path_to_load_chunks_words, f))]
for ind_file,file_name in enumerate(file_names):
    df2 = pd.read_pickle(os.path.join(path_to_load_chunks_words, file_name))
    print(type(df2), " : ", len(df2))
    #print("1. ***********************************")
    #print_structure_type(df2['line'])
    df2['line2'] = df2['line'].apply(tokenizer.encode_plus, return_tensors = 'pt', add_special_tokens = False, return_attention_mask = True)
    #flat_list = [item for sublist in df2['line2'] for item in sublist]
    #print(flat_list)
    input_ids = []
    attention_masks = []
    for iii in df2['line2']:
        #print(type(iii['input_ids']))
        #print((iii['input_ids'].numpy()))
        #print(type(iii['input_ids'].numpy()[0]))
        #print((iii['input_ids'].numpy()[0]))
        input_ids = np.concatenate((input_ids, iii['input_ids'].numpy()[0]), axis=None)
        attention_masks = np.concatenate((attention_masks, iii['attention_mask'].numpy()[0]), axis=None)
        #exit(0)
    
    print(len(input_ids), len(attention_masks))
    
    x = input_ids
    x = list(divide_chunks(x, chunk_size-2))
    del x[-1]
    if len(x) <= 0:
        print('*************len(x) <= 0******************')
        continue
    print(len(x), len(x[0]))
    x = np.insert(x, 0, 1, axis=1)
    x = np.insert(x, chunk_size-1, 2, axis=1).tolist()
    # <class 'list'> <class 'list'> 
    # <class 'list'> <class 'pandas.core.series.Series'>
    #print_structure_type(x)
    t = attention_masks
    t = list(divide_chunks(t, chunk_size-2))
    del t[-1]
    
    #print(t)
    print(len(t), len(t[0]))
    for i in range(len(t)):
        t[i] = np.append(t[i],[1,1]).tolist()
    print(len(t), len(t[0]))
    print(type(t), type(t[0]))
    
    df_chunks = pd.DataFrame({'chunks_ints': x, 'attention_masks': t})
    df_chunks['chunks_ints'] = df_chunks['chunks_ints'].apply(applay_list)
    #df_chunks['len_sent_ints'] = df_chunks['chunks_ints'].apply(len)
    df_chunks.to_pickle(path_to_save_chunks_ints + file_name)
    print(file_name)

    
    
