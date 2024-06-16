
import pandas as pd
import numpy as np

import os
import sys
from os import listdir
from os.path import isfile, join
import pickle
import json
from random import random, seed

from transformers import get_linear_schedule_with_warmup
# https://github.com/huggingface/transformers/issues/15212
# torchmetrics==0.6.2
    
import torch
device = torch.device("cpu")

import version_bert.word2vec_version.utility.variable as var
import version_bert.word2vec_version.utility.parameters as param2

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

model_path = 'C:/Users/User/source/repos/PythonApplication1/PythonApplication1/version_bert/BerelRun1_72580/'

path_to_chunks_ints = param2.save_delivery_here + "c2"+param2.version+"\\chunk_"+str(param2.chunk_size)+"\\"
path_to_dir_dict = param2.save_delivery_here + "d"+param2.version+"\\"
isExist = os.path.exists(path_to_dir_dict)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path_to_dir_dict)
  print("The new directory is created!")

threshold_year = 1800
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

min_dis = param2.min_distance
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

from transformers import BertModel, BertForMaskedLM
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32
max_examples = param2.max_examples
for threshold_year in year_thresholds:
    if threshold_year <= 1700: #660:
        continue
    #if threshold_year > 750: #1359:
    #    break
    
    print("**************************************************")
    print(" year: ", threshold_year)
    print("**************************************************")
    max_before = threshold_year-min_dis
    min_before = threshold_year-300
    min_after = threshold_year+min_dis
    max_after = threshold_year+300
    #df_with_target['target'] = df_with_target['year'].apply(update_target, t=threshold_year,
    #                             until_year = max_after,
    #                             min_year = min_before)
    df_with_target['target'] = df_with_target['year'].apply(update_target2, t=threshold_year)
    

    # calculate the prporation between the minimum set size and the maximum set size
    before_year = len(df_with_target[df_with_target['target'] == 0])
    after_year = len(df_with_target[df_with_target['target'] == 1])
    number_of_examples = min(before_year, after_year)
    print("*******************************************")
    print(before_year, after_year)
    print("*******************************************")
    
    if before_year == 0 or after_year == 0:
        continue
    
    select_from = 0
    if before_year < after_year:
        select_from = 1

    print("##########################################")
    print(number_of_examples)
    print("##########################################")

    if number_of_examples > max_examples:
        proportion = max_examples/max(before_year, after_year)
    else:
        proportion = number_of_examples/max(before_year, after_year)

    is_selected_set = df_with_target['target'].apply(select_random, s_f=select_from, por=proportion)
    current_df = df_with_target[is_selected_set==1]

    if number_of_examples > max_examples:
        select_from += 1
        select_from = select_from % 2
        proportion = max_examples/min(before_year, after_year)
        is_selected_set = current_df['target'].apply(select_random, s_f=select_from, por=proportion)
        current_df = current_df[is_selected_set==1]
        print("number_of_examples > max_examples")
    param2.epochs_num = 5
    if((after_year+before_year) >= 300000):
        param2.epochs_num = 4
    if((after_year+before_year) >= 450000):
        param2.epochs_num = 3
    if((after_year+before_year) >= 600000):
        param2.epochs_num = 2

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
    
    

    print("*************** A ******************")
    print(type(input_ids), type(attention_masks), type(labels))
    print(len(input_ids), len(attention_masks), len(labels))
    print("*************** B ******************")
    print(type(input_ids[0]), type(attention_masks[0]), type(labels[0]))
    print((input_ids[0].size(0)), (attention_masks[0].size(0)), "int_val: " + str(labels[0]))
    print("*************** C ******************")
    print(type(input_ids[0][0]), type(attention_masks[0][0]), "int_val: " + str(labels[0]))
    print("int_val: " + str(input_ids[0][0]), "int_val: " + str(attention_masks[0][0]), "int_val: " + str(labels[0]))
    print("*************** C ******************")
    print(type(input_ids[0][0]), type(attention_masks[0][0]), "int_val: " + str(labels[0]))
    print("int_val: " + str(input_ids[0]), "int_val: " + str(attention_masks[0]), "int_val: " + str(labels))

    #input_ids = torch.cat(input_ids, dim=0)
    #attention_masks = torch.cat(attention_masks, dim=0)
    
    print(input_ids.shape, attention_masks.shape, labels.shape)
    print(input_ids.size(0), attention_masks.size(0), labels.size(0))

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        model_path, 
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
   

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )


    # Number of training epochs. The BERT authors recommend between 2 and 4. 
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 1

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # training

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):
    
        # ========================================
        #               Training
        # ========================================
    
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 2 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
            
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            # https://github.com/huggingface/transformers/issues/2952
            b_input_ids = batch[0].type(torch.LongTensor)
            b_input_mask = batch[1].type(torch.LongTensor)
            b_labels = batch[2].type(torch.LongTensor)
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)
        
            #print("******************" + str(step) + "****************************")
            #print(len(b_labels))
            for ind_lab in range(len(b_labels)):
                if b_labels[ind_lab]==2:
                    print(b_labels[ind_lab])
                #if labels[ind_lab] == 2:
                #    print(ind_lab, labels[ind_lab])
            #print(len(current_df['target']>1))
            #print(len(current_df['target']==2))

            #print(current_df[current_df['target']>1])
            print("*****************************************************")
    
            
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # In PyTorch, calling `model` will in turn call the model's `forward` 
            # function and pass down the arguments. The `forward` function is 
            # documented here: 
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            # Specifically, we'll get the loss (because we provided labels) and the
            # "logits"--the model outputs prior to activation.
            result = model(b_input_ids, 
                           token_type_ids=None, 
                           attention_mask=b_input_mask, 
                           labels=b_labels,
                           return_dict=True)

            loss = result.loss
            logits = result.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
    
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
        
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].type(torch.LongTensor)
            b_input_mask = batch[1].type(torch.LongTensor)
            b_labels = batch[2].type(torch.LongTensor)
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)
        
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(b_input_ids, 
                               token_type_ids=None, 
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)

            # Get the loss and "logits" output by the model. The "logits" are the 
            # output values prior to applying an activation function like the 
            # softmax.
            loss = result.loss
            logits = result.logits
            
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
    
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
    
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



    import os

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    output_dir = './bert_model_save_min_'+str(min_dis)+'/' + str(threshold_year) + '/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    #tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

"""
###############################################
1486 159098 839488 317984
###############################################
*****************************************************
current_df['chunks_ints'].shape:  (317984,)
current_df['chunks_ints'][0].shape:  (407,)
current_df['chunks_ints'][1].shape:  (434,)
*****************************************************
###############################################
1554 169361 759845 338463
###############################################
*****************************************************
current_df['chunks_ints'].shape:  (338463,)
current_df['chunks_ints'][0].shape:  (431,)
current_df['chunks_ints'][1].shape:  (464,)
*****************************************************
"""