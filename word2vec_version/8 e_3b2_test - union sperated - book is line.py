
"""
Report test_set accuracy - create csv file, 
create table for each book (line) 
    what are the predicted min year and the predicted max year
Step 7 B
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from copy import deepcopy
from keras.models import load_model
import pickle
import os
import version_bert.word2vec_version.utility.parameters as param2

path_to_dir_test_set = param2.save_delivery_here + "d_test\\chunk_"+str(param2.chunk_size)+"\\chunks_by_book\\"
path_to_map_ind_to_filename = param2.save_delivery_here + "d_test\\"

map_all_results = {}
check_year = 1428
until_year = 1796 #1330 check_year
from_year =  260 #260 check_year
last_year = 1899 #1796 #1428 #1899

map_file_name='e2b_map_all_results_'+param2.pre_prefix_parameters+'_year_'+str(last_year)+'.p'
#map_file_name = 'e2b_map_all_results_from_year_'+str(from_year)+'_to_year_' +str(until_year)  +  '.p'
with open(os.path.join(path_to_dir_test_set + 'results\\', map_file_name), 'rb') as fp: # 1948
    map_all_results = pickle.load(fp)

map_years = {}
with open(os.path.join(path_to_dir_test_set, 'map_years.p'), 'rb') as fp:
    map_years = pickle.load(fp)
map_ind_to_filename={}
with open(os.path.join(path_to_map_ind_to_filename, 'map_ind_to_filename.p'), 'rb') as fp:
    map_ind_to_filename = pickle.load(fp)



import version_bert.word2vec_version.utility.a1_data as data
year_thresholds = data.year_books_thresholds

tresh = 50
isWighted = True
strline = "book_id, real_year, chunks, sum_to_0, sum_to_1, sum_from_0, sum_from_1, from_year, to_year"
with open(path_to_dir_test_set + "results\\" + "e3b2_test_set_dif_books_all_acc_"+str(tresh)+"_"+str(isWighted)+"_"+param2.pre_prefix_parameters+".csv", "w", encoding="utf-8") as output_file: 
    output_file.write(strline+"\n")


all_books = {}
for ind in range(param2.test_set_size):
    #if map_years[ind] < 1000:
    #    continue
    print(ind, str(map_years[ind]), map_ind_to_filename[ind])
    strline = str(ind) #map_ind_to_filename[ind]
    strline += ", " + str(map_years[ind])
    predict_from_year = -600
    predict_to_year = 2023
    save_all_cases = 0
    sum_0_to = 0
    sum_0_from = 0
    sum_1_to = 0
    sum_1_from = 0
    for year in year_thresholds:
        if year <= 260: #260: #261: 1948 # 1674
            continue
        #if year < check_year:
        #    continue;
        #if year > check_year:
        #    break;
        if year < from_year:
            continue
        if year > until_year:
            break
        if year not in map_all_results.keys():
            print("year not in map_all_results.keys()")
            continue
        if ind not in map_all_results[year].keys():
            print("ind not in map_all_results[year].keys()")
            continue

        res = map_all_results[year][ind]
        np_true_labels = res['np_true_labels']
        np_predictions = res['np_predictions']
        
        # calculating acc
        np_predictions_arg_max = np.argmax(np_predictions, axis=1)
        sum_0 = 0
        sum_1 = 0
        for item in np_predictions:
            sum_0 += item[0]
            sum_1 += item[1]
        
        #print(np_predictions_arg_max)
        count_zeros_cases = np.count_nonzero(np_predictions_arg_max == 0)
        count_ones_cases = np.count_nonzero(np_predictions_arg_max == 1)
        all_cases = count_zeros_cases+count_ones_cases
        acc = count_ones_cases / all_cases
        if save_all_cases == 0:
            save_all_cases = all_cases
        #print("*****************Summery*******************")
   
        #print(year, ind, acc)
        """
        if (acc<=tresh/100):
            if predict_to_year == 2023:
                print("predict_to_year: ", predict_to_year)
                predict_to_year = year
        if (acc>=1-tresh/100):
            predict_from_year = year
        """
        if (sum_1<=sum_0):
            if predict_to_year == 2023:
                print("predict_to_year: ", predict_to_year)
                predict_to_year = year
                sum_0_to = sum_0
                sum_1_to = sum_1
        if (sum_1>=sum_0):
            predict_from_year = year
            sum_0_from = sum_0
            sum_1_from = sum_1
        
    
    print(predict_from_year, predict_to_year)      
    strline += ", " + str(save_all_cases) + ", " + str(sum_0_to) + ", " + str(sum_1_to) + ", " + str(sum_0_from) + ", " + str(sum_1_from) + ", " + str(predict_from_year) + ", " + str(predict_to_year)    
    with open(path_to_dir_test_set + "results\\" + "e3b2_test_set_dif_books_all_acc_"+str(tresh)+"_"+str(isWighted)+"_"+param2.pre_prefix_parameters+".csv", "a", encoding="utf-8") as output_file: 
        output_file.write(strline + "\n")
    