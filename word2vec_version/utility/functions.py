import pandas as pd
def print_structure_type(ob, count= 0):
    print("In count: ", count," type: ", type(ob))
    
    if type(ob) == str:
        print(" str count: "," len: ", len(ob), count," value: ",ob[0])
    elif type(ob) == int:
        print(" int count: ", count," value: ",ob)
    elif type(ob) == list:
        print(" list count: ", count," type: ", type(ob), " len: ", len(ob))
        if len(ob) > 0:
            print_structure_type(ob[0], 1+count)
    elif hasattr(ob, '__shape__'):
        print(" shape( count: ", count," type: ", type(ob), " shape: ", ob.shape)
        if ob.shape[0] > 0:
            print_structure_type(ob[0], 1+count)
    elif isinstance(ob, pd.core.series.Series):
        ob = ob.tolist()
        print(" Series_to_list( count: ", count," type: ", type(ob), " len: ", len(ob))
        if len(ob) > 0:
            print_structure_type(ob[0], 1+count)
    elif hasattr(ob, '__len__'): 
        print(" hasattr( count: ", count," type: ", type(ob), " len: ", len(ob))
    else:
        print(" count: ", count," value: ",ob)

def always_continue_func(t_year, year):
    return False

import version_bert.word2vec_version.utility.parameters as param
save_delivery_here = param.save_delivery_here
def save_book(df_dummy, i, name, year, num_of_words, x):
    str_line = "\"" + name + "\", " + str(year)+ ", " + str(num_of_words)
    #print(str_line)
    with open(save_delivery_here + "books_analysis.csv", "a", encoding="utf-8") as output_file: 
        output_file.write(str_line + "\n")
    

def update_year_ranges(year_ranges, i, name, year, num_of_words, x):
    if year not in year_ranges.keys():
        year_ranges[year] = num_of_words
    else:
        year_ranges[year] += num_of_words

def in_range_continue_func(t_year, year):
    max_before = t_year-25
    min_before = t_year-300
    min_after = t_year+25
    max_after = t_year+300
    if int(year) > min_after and int(year)<=max_after or int(year) >= min_before and int(year)<max_before:
        return False
    return True

def update_word_freq(word_freq, i, name, year, num_of_words, x):
    for word in x:
        if word in word_freq.keys():
            word_freq[word] += 1
        else:
            if len(word) > 0:
                word_freq[word] = 1


path_to_save_data_frame = param.save_delivery_here + "c"+param.version+"\\chunk_"+str(param.chunk_size)+"\\"
def update_data_frame(df_dummy, i, name, year, num_of_words, x):
    df_chunks = pd.DataFrame({'x': x})
    df_chunks['target'] = year

    df_chunks.to_pickle(path_to_save_data_frame + str(year) + '_df_' + str(i) + '_without_chunks_ints.p')
    print('{}_df_{}_without_chunks_ints.p was saved'.format(year, i))

path_to_save_clean_data = param.save_delivery_here + "a\\"
def save_clean_data(df_dummy, i, name, year, num_of_words, x):
    with open(path_to_save_clean_data+name, 'w') as f:
        f.write(' '.join(x))