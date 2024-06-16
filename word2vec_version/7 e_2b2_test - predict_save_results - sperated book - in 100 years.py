"""
Save pickle file for all results for each year and for each book
Step 6
"""
import torch
device = torch.device("cpu")

from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from copy import deepcopy

import pickle
import os

import version_bert.word2vec_version.utility.parameters as param2

#path_to_dir_models = param2.data_dir + "version_2\\" + "d\\"
#path_to_dir_test_set = param2.data_dir + "version_3\\" + "d_test\\"

import version_bert.word2vec_version.utility.a1_data as data
year_thresholds = data.year_books_thresholds

until_year = 2020 #1330
from_year = 0 #260

path_to_dir_test_set = param2.save_delivery_here + "d_test\\chunk_"+str(param2.chunk_size)+"\\chunks_by_book\\"

map_all_results = {}

#file_name = 'map_all_results_'+param2.pre_prefix_parameters+'_year_'+str(1899)+'.p'
#with open(os.path.join(path_to_dir_test_set, file_name), 'rb') as fp:
#    map_all_results = pickle.load(fp)

map_years = {}
map_df = {}
for ind in range(param2.test_set_size):
    base_file_name = str(ind)+"_model_number_books_5_16_train_X_test.p"
    current_df_path = path_to_dir_test_set + base_file_name
    with open(current_df_path, 'rb') as fp:
        current_df = pickle.load(fp)
    map_years[ind] = current_df['year'][0]
    map_df[ind] = current_df
    print(ind, current_df_path, len(current_df))

with open(os.path.join(path_to_dir_test_set, 'map_years.p'), 'wb') as fp:
    pickle.dump(map_years, fp, protocol=pickle.HIGHEST_PROTOCOL)


def update_target(x, t):
    if x<t-(param2.min_distance-1):
        return 0;
    if x>=t+(param2.min_distance-1):
        return 1;
    return 2;


def select_all(x):
    if x == 2:
        return 0;
    return 1

def select_in(x):
    if x == 2:
        return 1;
    return 0


base_file_name = param2.prefix_name
#base_file_name = "model_number_books_5_16_train"
previous_year = -1
batch_size = 32
for year in year_thresholds:
    print(year)
    if year <= 260: #660     #260: #261: 1948 # 1674
       continue
    #if year > 660: 
    #    break
    if year <= from_year:
        continue;
    if year > until_year:
        break;

    output_dir = './bert_model_save/'+str(year)+'/'
    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForSequenceClassification.from_pretrained(output_dir)
    model.to(device)

    
    map_all_results[year] = {}

    for ind in range(param2.test_set_size):
        df_with_target = map_df[ind]
        df_with_target['target'] = df_with_target['year'].apply(update_target, t=year)    
        before_year = len(df_with_target[df_with_target['target'] == 0])
        after_year = len(df_with_target[df_with_target['target'] == 1])
    
        is_selected_set = df_with_target['target'].apply(select_in)
        current_df = df_with_target[is_selected_set==1]
        
        print("index: " + str(ind),len(current_df['target']), len(current_df['chunks_ints']), len(current_df['attention_masks']))
        if len(current_df['target']) == 0:
            continue
        #if ind == 109:
        #    continue
        labels = current_df['target'].to_numpy()
        labels = torch.tensor(labels)
        attention_masks = current_df['attention_masks'].tolist()
        input_ids = current_df['chunks_ints'].tolist()
        attention_masks = torch.as_tensor([(x) for x in attention_masks])
        input_ids = torch.as_tensor([(x) for x in input_ids])


        # Create the DataLoader.
        prediction_data = TensorDataset(input_ids, attention_masks, labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


        # Prediction on test set
        print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

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
        
        # calculating acc
        np_predictions_arg_max = np.argmax(np_predictions, axis=1)
        true_prediction = np_predictions_arg_max == np_true_labels
    
        currect_prediction = np.count_nonzero(true_prediction == True)
        print(true_prediction.sum())
        print(currect_prediction)
    
        print("*****************Summery*******************")
        acc =(currect_prediction / len(true_prediction) * 100.0) 
        print('Positive samples: %d of %d (%.2f%%)' % (currect_prediction, len(true_prediction), acc))
        #if acc > 100:
        print(acc)
        map_all_results[year][ind] = {
             'np_true_labels': np_true_labels, 
             'np_predictions': np_predictions
             }
    
    file_name = 'e2b_map_all_results_'+param2.pre_prefix_parameters+'_year_'+str(year)+'_in_100.p'

    with open(os.path.join(path_to_dir_test_set + 'results\\', file_name), 'wb') as fp:
        pickle.dump(map_all_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    file_name = 'e2b_map_all_results_'+param2.pre_prefix_parameters+'_year_'+str(previous_year)+'_in_100.p'
    if previous_year > 0:
        os.remove(os.path.join(path_to_dir_test_set+ 'results\\', file_name))
    
    previous_year = year