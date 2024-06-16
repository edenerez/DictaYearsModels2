"""
This file run the code that response 
about creating clean text files for each original text file
Step 1
"""
from version_bert.word2vec_version.utility.functions import update_data_frame, always_continue_func
import version_bert.word2vec_version.utility.text as txt
import os
import version_bert.word2vec_version.utility.parameters as param

path_to_save_data_frame = param.save_delivery_here + "c"+param.version+"\\chunk_"+str(param.chunk_size)+"\\"
isExist = os.path.exists(path_to_save_data_frame)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path_to_save_data_frame)
  print("The new directory is created!")

df_dummy = dict()
txt.csv_iterations(update_data_frame, df_dummy, always_continue_func, 0)

        # move this step to the next step at c_convert_chunk_words_t_ints
        # because of moving this step to the next step
        # there was an error since the chunks at the next step 
        # creates list of list of word 
        # instead of list of words
        #x = list(divide_chunks(x, chunk_size))
        #del x[-1]

        
        #df_chunks = pd.DataFrame({'x': x})
        
        #df_chunks['target'] = year
        #df_chunks.to_pickle(path_to_save_chunks + str(year) + '_df_' + str(i) + '_without_chunks_ints.p')
        #print('{}_df_{}_without_chunks_ints.p was saved'.format(year, i))

    


