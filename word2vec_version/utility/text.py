

import unicodedata
from os.path import isfile, join
import sys
import pandas as pd
import version_bert.word2vec_version.utility.variable as var
import version_bert.word2vec_version.utility.parameters as param

word2idx = var.word2idx
df = var.csv_reader

def select_sentence(sentence):
    if '###' in sentence:
        return False
    elif len(sentence) < 5:
        return False
    else:
        return True

def remove_vowels(sentence):
    nikudnik = str(sentence)
    normalized = unicodedata.normalize('NFKD', nikudnik)  # Remove hebrew vowel ניקוד marks
    flattened = ''.join([c for c in normalized if not unicodedata.combining(c)])
    return flattened


def remove_spacial_characters(sentence):
    sentence = sentence.replace('\n', '')
    sentence = sentence.replace(':', '')
    return sentence


def remove_word_not_exist_in_dict(sentence):
    word_list = []
    words = sentence.split()
    for word in words:
        if word in word2idx:
            word_list.append(word)
    new_sentence = ' '.join(word_list)
    #print(new_sentence)
    return new_sentence


def _not_used__remove_stars(sentence):
    if '**' in sentence:
        return False
    else:
        return True

def csv_iterations(call_func, param_to_update, not_continue_func, t_year):
    path_to_save_data_frame = param.save_delivery_here + "c"+param.version+"\\chunk_"+str(param.chunk_size)+"\\"

    for i in df.index:
        if i > param.train_number:
            break
        name = df[str("filename")][i]
        if len(name) == 0:
            break
        year = df[str("year")][i]
        if(not_continue_func(t_year, year)):
            continue
        try:
            
            #if year not in year_ranges.keys():
            #    year_ranges[year] = 0
            
            input_file = open(param.file_dir + name, 'r', encoding="utf-8")
            all_lines = input_file.readlines()
            input_file.close()
            df2 = pd.DataFrame({'line': all_lines, 'source_file': [name] * len(all_lines)})
            
            df2 = df2[df2['line'].apply(select_sentence)]
            ## df2['line'] = df2['line'].apply(remove_vowels)
            ## df2['line'] = df2['line'].apply(remove_spacial_characters)
            
            # comment if need to check without removing not existing words
            ### df2['line'] = df2['line'].apply(remove_word_not_exist_in_dict)
            
            #df['target'] = df['source_file'].apply(year)

            num_of_words = len(' '.join(df2['line'].values).split(' '))
            print(i, name, year, num_of_words)
            
            df2['target'] = year

            df2.to_pickle(path_to_save_data_frame + str(year) + '_df_' + str(i) + '_without_chunks_ints.p')
            print('{}_df_{}_without_chunks_ints.p was saved'.format(year, i))
            ###call_func(param_to_update, i, name, year, num_of_words, x)
            
            #year_ranges[year] += num_of_words
            str_line = name + ", " + str(year)+ ", " + str(num_of_words)
        
            #with open(path_output_file, "a", encoding="utf-8") as output_file: 
            #    output_file.write(str_line + "\n")
        
        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
