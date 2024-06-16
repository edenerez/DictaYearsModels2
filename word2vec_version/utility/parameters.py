import version_bert.word2vec_version.utility._parameters as variables

import warnings
warnings.filterwarnings("ignore")

version = '_train' #'_train' #'_test'
train_number = 1240 

max_examples = variables.max_examples
dict_at_least_words = variables.dict_at_least_words  #3 #2
chunk_size = variables.chunk_size 
epochs_num = variables.epochs_num # 5 - 2
dropout = variables.dropout # 0.1 - 0.3
batch_size = variables.batch_size # 128
number_of_units = variables.number_of_units # 100
num_of_layers = variables.num_of_layers
optimizer = variables.optimizer['value']

base_path = 'C:\\Users\\User\\source\\repos\\'
project_dir = base_path+"PythonApplication1\\PythonApplication1\\"
data_dir = project_dir + 'data_3.7\\' # last version: 'data_3.7_books\\' # old: 'data_3.7\\'

csv_file_path = "C:\\Users\\User\\source\\repos\\PythonApplication1\\PythonApplication1\\data_3.7\\version_4_Final_DatesForTheCorpus"+version+".csv"
#csv_file_path = "C:\\Users\\User\\source\\repos\\PythonApplication1\\PythonApplication1\\data_3.7\\‏‏‏‏FullR25_Responsa_DatesForTheCorpus_pshut_dates_wip_train.csv"
#csv_file_path = "C:\\Users\\User\\source\\repos\\PythonApplication1\\PythonApplication1\\data_3.7\\FullR25_Responsa_DatesForTheCorpus_pshut_dates_wip_test.csv"
#csv_file_path = data_dir+'FullR25_Responsa_DatesForTheCorpus_pshut_dates_wip' + version + '.csv'

file_dir = project_dir+'ResponsaRedividedByMR\\'

dictioanry_file_name = 'a_vocabulary_dict.json'
dictioanry_freq_file_name = 'vocabulary_word_freq.json'

path_to_save = data_dir+'vocabulary\\'
path_to_load_dict = path_to_save + 'a3_vocabulary_dict_at_least_'+str(dict_at_least_words)+'.json'
filtered_dict_file_name = 'vocabulary_dict_at_least_'+str(dict_at_least_words)+'.json'

# version 5 - 
# דוגמאות דוגמים מלפני 75 שנה ואחרי 75 שנה מהשנה שאותה בודקים
# לוקחים דוגמאות אימון בטווח של 300 שנה
# לקחת פחות דוגמאות בשביל לאמן מהר
# המילון יורכב ממילים שהופיע לפחות 20, 30 או 40 הופעות
# לקחת מספר שווה של דוגמאות אימון מלפני ומאחריי אותה שנה שבודקים
# שיהיה אפשר להמשיך לאמן את ה WORD2VEC
# גודל צ'אנק יהיה 400 מילים
# לעשות ניתוח לכל מילה באיזו שנה הופיע לראשונה ובאיזו שנה הופיע בפעם האחרונה

save_delivery_here = data_dir + "version_bert\\word2vec_version\\" #"version_3\\"


w2v_count = dict_at_least_words
w2v_vec_len = variables.w2v_vec_len #200
#w2v_window_size = 5
#path_to_load_gensim = path_to_save + 'c_gensim_word2vec_'+str(dict_at_least_words)+'_'+str(w2v_vec_len)+'.bin'
#path_to_load_gensim = path_to_save + 'd_gensim_word2vec_train_'+str(dict_at_least_words)+'_'+str(w2v_vec_len)+'.bin'
only_exist_words = variables.only_exist_words
min_distance = variables.min_distance
pre_prefix_parameters = 'examples_' + str(max_examples) + '_emdedding_'+str(w2v_vec_len)+'_dis_'+str(min_distance)+'_chunk_'+ str(chunk_size) + '_dropout_' + str(dropout)+ '_opt_' + str(variables.optimizer['name'])
if number_of_units != 100:
    pre_prefix_parameters = 'examples_' + str(max_examples) + '_emdedding_'+str(w2v_vec_len)+'_dis_'+str(min_distance)+'_chunk_'+ str(chunk_size) + '_dropout_' + str(dropout)+ '_opt_' + str(variables.optimizer['name'])
    #pre_prefix_parameters = 'num_of_layers_' + str(num_of_layers) + '_number_of_units_' + str(number_of_units) + '_' + pre_prefix_parameters
    pre_prefix_parameters = 'units_' + str(number_of_units) + '_' + pre_prefix_parameters
    if num_of_layers != 1:
        pre_prefix_parameters = 'layers' + str(num_of_layers) + '_' + pre_prefix_parameters
    
    prefix_name = pre_prefix_parameters+'_model_5_16_train'
    
else:
    prefix_name = pre_prefix_parameters+'_model_number_books_5_16_train'

#path_to_load_gensim = path_to_save + 'd_gensim_word2vec_train_'+str(dict_at_least_words)+'_'+str(w2v_vec_len)+'.bin'

test_set_size = 111