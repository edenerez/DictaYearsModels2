"""
Report test_set accuracy for each year
Step 5
"""
import torch
device = torch.device("cpu")

from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from copy import deepcopy

from keras.models import load_model
import pickle
import os
import version_bert.word2vec_version.utility.parameters as param2

path_to_dir_test_set = param2.save_delivery_here + "d_test\\chunk_"+str(param2.chunk_size)+"\\all_chunks\\"

max_examples = param2.max_examples
file_name = "model_number_books_5_16_train_X_test.p"

with open(path_to_dir_test_set + file_name, 'rb') as fp:
    df_with_target = pickle.load(fp)

import version_bert.word2vec_version.utility.a1_data as data
year_thresholds = data.year_books_thresholds
until_year = 1701 #2020 #1328 #1751 #1840 #1330
from_year = 0 #1260 #1330 #260

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

map_all_results = {}
full_file_name = path_to_dir_test_set + "e2_test_set_dif_books_"+param2.pre_prefix_parameters+"_acc.csv"
with open(full_file_name, "w", encoding="utf-8") as output_file: 
    output_file.write("year, test_set_acc, tn, fp, fn, tp\n")

batch_size = 32
for year in year_thresholds:
    print(year)
    if year < 760: #1428: #260: #261: 1948 # 1674
        continue
    #if year > 1428: 
    #    break;
    if year < from_year:
        continue;   
    if year > until_year:
        break;

    output_dir = './bert_model_save/'+str(year)+'/'
    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForSequenceClassification.from_pretrained(output_dir)
    model.to(device)

    df_with_target['target'] = df_with_target['year'].apply(update_target, t=year)    
    
    before_year = len(df_with_target[df_with_target['target'] == 0])
    after_year = len(df_with_target[df_with_target['target'] == 1])
    is_selected_set = df_with_target['target'].apply(select_all)
    current_df = df_with_target[is_selected_set==1]
    
    print(len(current_df['target']), len(current_df['chunks_ints']), len(current_df['attention_masks']))

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
    
    file_name = 'e2_labels_'+param2.pre_prefix_parameters+'_year_'+str(year)+'.p'
    with open(os.path.join(path_to_dir_test_set, file_name), 'wb') as fp:
        pickle.dump(np_true_labels, fp, protocol=pickle.HIGHEST_PROTOCOL)
    file_name = 'e2_predictions_'+param2.pre_prefix_parameters+'_year_'+str(year)+'.p'
    with open(os.path.join(path_to_dir_test_set, file_name), 'wb') as fp:
        pickle.dump(np_predictions, fp, protocol=pickle.HIGHEST_PROTOCOL)

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
    
    #print("Shape: " + str(res.shape) + " first result: " + str(res[0,:].sum()))
    map_all_results[year] = acc

    file_name = 'e2_map_all_results_'+param2.pre_prefix_parameters+'_year_'+str(year)+'.p'
    with open(os.path.join(path_to_dir_test_set, file_name), 'wb') as fp:
        pickle.dump(map_all_results, fp, protocol=pickle.HIGHEST_PROTOCOL)

    """
    max_pred = np.zeros(len(res))
    for i in range(len(res)):
        max_pred[i] = np.argmax(res[i])
    max_pred = [int(x) for x in max_pred]

    y_test_without_dummies = list(y_test[y_test==1].stack().reset_index().drop(0,1)['level_1'])
    
    
    #import seaborn as sns
    tn, fp, fn, tp = confusion_matrix(y_test_without_dummies, max_pred, labels=[0, 1]).ravel()

    labels_dict = {
        0: 'Before ' + str(year),
        1: 'After ' + str(year)
    }

    y_test_and_pred_df = pd.DataFrame({'test': y_test_without_dummies, 'pred': max_pred})

    y_test_and_pred_df['test_str'] = y_test_and_pred_df['test'].map(labels_dict)
    y_test_and_pred_df['is_correct'] = np.where(y_test_and_pred_df['test'] == y_test_and_pred_df['pred'], 1, 0)

    y_test_and_pred_df['test_str'].value_counts()

    y_test_and_pred_df['test_str'].value_counts(normalize=True)
    print("model year: " + str(year))
    print(f'Accuracy score: {np.mean(y_test_and_pred_df.is_correct)}')
    
    str_line = str(year) + ", " + str(f'{np.mean(y_test_and_pred_df.is_correct)}')+ ", " + str(tn) + ", " + str(fp) + ", " +str(fn)+ ", " +str(tp)
    with open(full_file_name, "a", encoding="utf-8") as output_file: 
        output_file.write(str_line + "\n")
    """
print ("results are saved to: " + full_file_name)