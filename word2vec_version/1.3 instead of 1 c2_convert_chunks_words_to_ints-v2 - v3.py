

from version_bert.word2vec_version.chunks.module import applay_list, tokenizer
from version_bert.word2vec_version.chunks.module import path_to_load_chunks_words 
from version_bert.word2vec_version.chunks.module import divide_chunks, process_df
from version_bert.word2vec_version.chunks.module import process_input_ids, process_attention_masks, save_df


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

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []
chunk_size = param2.chunk_size
file_names = [f for f in listdir(path_to_load_chunks_words) if isfile(join(path_to_load_chunks_words, f))]
for ind_file,file_name in enumerate(file_names):
    df2 = pd.read_pickle(os.path.join(path_to_load_chunks_words, file_name))
    
    input_ids, attention_masks = process_df(df2)
    
    x = process_input_ids(input_ids)
    if x is None:
        continue
    # <class 'list'> <class 'list'> 
    # <class 'list'> <class 'pandas.core.series.Series'>
    #print_structure_type(x)
    
    t = process_attention_masks(attention_masks)
    
    save_df(x, t, file_name)
    

    
    
