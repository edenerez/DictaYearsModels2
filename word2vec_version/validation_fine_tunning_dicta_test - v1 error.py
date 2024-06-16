import os
import sys
from os import listdir
from os.path import isfile, join
import pickle
import json
from random import random, seed

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch
import pandas as pd
import numpy as np

device = torch.device("cpu")


from transformers import BertForSequenceClassification
from transformers import BertTokenizer
threshold_year = 1302
output_dir = './bert_model_save/'+str(threshold_year)+'/'

# Load a trained model and vocabulary that you have fine-tuned
model = BertForSequenceClassification.from_pretrained(output_dir)

# Copy the model to the GPU.
model.to(device)

# testing
import pandas as pd

import version_bert.word2vec_version.utility.variable as var
import version_bert.word2vec_version.utility.parameters as param2

path_to_chunks_ints = param2.save_delivery_here + "c2"+param2.version+"\\chunk_"+str(param2.chunk_size)+"\\"
path_to_dir_dict = param2.save_delivery_here + "d"+param2.version+"\\"
isExist = os.path.exists(path_to_dir_dict)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path_to_dir_dict)
  print("The new directory is created!")


print("path_to_chunks_ints: " , path_to_chunks_ints)
file_names = [f for f in listdir(path_to_chunks_ints) if isfile(join(path_to_chunks_ints, f))]

df_with_target = pd.DataFrame(columns=['chunks_ints', 'attention_masks', 'year'])
count = 0
for file_name in file_names:
    index_of_underscore = file_name.find('_')
    str_target_year = file_name[:index_of_underscore]
    int_target_year = int(str_target_year)
    #if int_target_year >1000:
    #    break
    target = 0
    if(int_target_year>threshold_year):
        target=1
    count+=1
    #if(count>120):
    #    break
    print(str(len(df_with_target)) + "   " + str(count) + " year: " + str_target_year +" filename: " + file_name)
    try:
        df_temp = pd.read_pickle(os.path.join(path_to_chunks_ints, file_name))
        df_temp['year'] = [int_target_year] * len(df_temp['chunks_ints'])
        #attention_masks = df_temp['attention_masks']
        #chunks_ints = df_temp['chunks_ints']
        #labels = [target] * len(df_temp['chunks_ints'])
        df_with_target = pd.concat([df_temp, df_with_target[['chunks_ints', 'attention_masks', 'year']]])
    except:
        print("Unexpected error:", sys.exc_info()[0])
        continue

min_dis = 0 #param2.min_distance
def update_target2(x, t):
    if x<t-min_dis:
        return 0;
    if x>t+min_dis:
        return 1;
    return 2;

def update_target(x, t, until_year, min_year):
    if x<t-min_dis and x>=min_year:
        return 0;
    if x>t+min_dis and x<=until_year:
        return 1;
    return 2;

def select_random(x, s_f, por):
    if x == 2:
        return 0;
    if x == s_f:
        if random() <= por:
            return 1
        else:
            return 0
    return 1

import version_bert.word2vec_version.utility.a1_data as data
year_thresholds = data.year_books_thresholds
batch_size = 32
# *********************************************
# for each year
# *********************************************
df_with_target['target'] = df_with_target['year'].apply(update_target2, t=threshold_year)
# calculate the prporation between the minimum set size and the maximum set size
before_year = len(df_with_target[df_with_target['target'] == 0])
after_year = len(df_with_target[df_with_target['target'] == 1])
number_of_examples = min(before_year, after_year)
print("*******************************************")
print(before_year, after_year)
print("*******************************************")
    
#if before_year == 0 or after_year == 0:
#    continue
    
select_from = 0
if before_year < after_year:
    select_from = 1

print("##########################################")
print(number_of_examples)
print("##########################################")


current_df = df_with_target


print("###############################################")
print(threshold_year, before_year, after_year, len(current_df))
print("###############################################")

print("*****************************************************")
print("current_df['chunks_ints'].shape: " , current_df['chunks_ints'].shape)
print("current_df['chunks_ints'][0].shape: " , current_df['chunks_ints'][0].shape)
print("current_df['chunks_ints'][1].shape: " , current_df['chunks_ints'][1].shape)
print("*****************************************************")
    
print(len(current_df['target']), len(current_df['chunks_ints']), len(current_df['attention_masks']))

labels = current_df['target'].to_numpy()
labels = torch.tensor(labels)
attention_masks = current_df['attention_masks'].tolist()
input_ids = current_df['chunks_ints'].tolist()
attention_masks = torch.as_tensor([(x) for x in attention_masks])
input_ids = torch.as_tensor([(x) for x in input_ids])

back_up_labels = labels
back_up_attention_masks = attention_masks
back_up_input_ids = input_ids

for index_1000 in range(0, 100): #(10,11): # 930, 931
    from_index = 1000*index_1000
    to_index = from_index+1000

    labels = back_up_labels[from_index:to_index]
    attention_masks = back_up_attention_masks[from_index:to_index]
    input_ids = back_up_input_ids[from_index:to_index]
    
    """
    print("*************** A ******************")
    print(type(input_ids), type(attention_masks), type(labels))
    print(len(input_ids), len(attention_masks), len(labels))
    print("*************** B ******************")

    print(type(input_ids[0]), type(attention_masks[0]), type(labels[0]))

    print((input_ids[0].size(0)), (attention_masks[0].size(0)), "int_val: " + str(labels[0]))
    print(len(input_ids[0]), len(attention_masks[0]), "int_val: " + str(labels[0]))

    for i in range(50):
        if len(input_ids[i]) != 400:
            print(i)
    
    print("*************** C ******************")
    print(type(input_ids[0][0]), type(attention_masks[0][0]), "int_val: " + str(labels[0]))
    print("int_val: " + str(input_ids[0][0]), "int_val: " + str(attention_masks[0][0]), "int_val: " + str(labels[0]))
    print("*************** C ******************")
    print(type(input_ids[0][0]), type(attention_masks[0][0]), "int_val: " + str(labels[0]))
    print("int_val: " + str(input_ids[0]), "int_val: " + str(attention_masks[0]), "int_val: " + str(labels))
    """
    #input_ids = torch.cat(input_ids, dim=0)
    #attention_masks = torch.cat(attention_masks, dim=0)
    """
    print(input_ids.shape, attention_masks.shape, labels.shape)
    print(input_ids.size(0), attention_masks.size(0), labels.size(0))
    """

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


    # Prediction on test set
    print(index_1000, 'Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in prediction_dataloader:
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
  
      # Unpack the inputs from our dataloader
      #b_input_ids, b_input_mask, b_labels = batch
      b_input_ids = batch[0].type(torch.LongTensor)
      b_input_mask = batch[1].type(torch.LongTensor)
      b_labels = batch[2].type(torch.LongTensor)
      #b_input_ids = b_input_ids.to(device)
      #b_input_mask = b_input_mask.to(device)
      #b_labels = b_labels.to(device)

      # Telling the model not to compute or store gradients, saving memory and 
      # speeding up prediction
      with torch.no_grad():
          # Forward pass, calculate logit predictions.
          result = model(b_input_ids, 
                         token_type_ids=None, 
                         attention_mask=b_input_mask,
                         return_dict=True)

      logits = result.logits

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
  
      # Store predictions and true labels
      #print("predictions.append: " + str(len(logits)))
      predictions.append(logits)
      true_labels.append(label_ids)
      #np_predictions = np.concatenate(predictions)
      #print(len(predictions), len(np_predictions), np_predictions.sum())


    print('    DONE.')
    
    np_true_labels = np.concatenate(true_labels)
    np_predictions = np.concatenate(predictions)
    np_predictions_arg_max = np.argmax(np_predictions, axis=1)
    true_prediction = np_predictions_arg_max == np_true_labels
    
    currect_prediction = np.count_nonzero(true_prediction == True)
    print(true_prediction.sum())
    print(currect_prediction)
    
    print("*****************Summery*******************")
    acc =(currect_prediction / len(true_prediction) * 100.0) 
    print('Positive samples: %d of %d (%.2f%%)' % (currect_prediction, len(true_prediction), acc))
    #if acc > 100:
    print(acc, index_1000)
# Positive samples: 116891 of 100006 (116.88%)
# year=1302 false_examples=5695 
#            true_examples=71731 
#                      all=100006