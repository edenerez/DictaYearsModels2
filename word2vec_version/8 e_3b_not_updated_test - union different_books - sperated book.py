"""
Report test_set accuracy - create csv file, 
create accurcies table for each book (line) and for each year (column)
Step 7 A
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
map_all_results = {}

check_year = 1750 #1428
last_year = 1899 #1428 #1899

map_file_name='e2b_map_all_results_'+param2.pre_prefix_parameters+'_year_'+str(last_year)+'.p'
with open(os.path.join(path_to_dir_test_set + "results\\", map_file_name), 'rb') as fp: # 1948
    map_all_results = pickle.load(fp)


map_years = {}
with open(os.path.join(path_to_dir_test_set, 'map_years.p'), 'rb') as fp:
    map_years = pickle.load(fp)


import version_bert.tokenize_version.utility.a1_data as data
year_thresholds = data.year_books_thresholds
until_year = 1990 #1990 #1330
from_year = 260 #260 #260

is_wieght = False
acc = 20

strline = "year"
for ind in range(param2.test_set_size):
    #strline += ", " + str(ind) + "_" + str(map_years[ind])
    strline += ", " + str(map_years[ind])
    #strline += ", " + str(ind) + "_" + str(map_years[ind]) + "_tn"
    #strline += ", " + str(ind) + "_" + str(map_years[ind]) + "_fp"  
    #strline += ", " + str(ind) + "_" + str(map_years[ind]) + "_fn"
    #strline += ", " + str(ind) + "_" + str(map_years[ind]) + "_tp"

with open(path_to_dir_test_set + "results\\" + "e3b_test_set_dif_books_all_last_year_"+str(last_year)+"_"+param2.pre_prefix_parameters+".csv", "w", encoding="utf-8") as output_file: 
    output_file.write(strline+"\n")

all_books = {}

for year in year_thresholds:
    print(year)
    if year <= 260: #260: #261: 1948 # 1674
       continue
    if year < check_year:
        continue;
    if year > check_year:
        break;

    #if year < from_year:
    #    continue;
    #if year > until_year:
    #    break;
    print(year)
    strline = str(year)
    for ind in range(param2.test_set_size):
        #print(strline)
        if year not in map_all_results.keys():
            strline += ", "
        elif ind not in map_all_results[year]:
            strline += ", "
        else:
            acc = map_all_results[year][ind]
            strline += ", " + str(acc) #+ str(r)
            #strline += ", " + str(c_v)
    full_path = path_to_dir_test_set + "results\\" + "e3b_test_set_dif_books_all_last_year_"+str(last_year)+"_"+param2.pre_prefix_parameters+".csv"
    print(full_path)    
    with open(full_path, "a", encoding="utf-8") as output_file: 
        output_file.write(strline + "\n")