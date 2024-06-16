
only_exist_words = False
dict_at_least_words = 30 
chunk_size = 400 #400 
min_distance = 34 #60 #34  # 100
max_examples = 600000
emdedding_size = w2v_vec_len = 200 
epochs_num = 5 # 2
dropout = 0.3 # 0.1
batch_size = 16 # 16 128
number_of_units = 100 #400 # 100
num_of_layers = 1 # 3 64. 2 64
import keras
optimizer = {'name': "Adam" , 'value': keras.optimizers.Adam(learning_rate=1e-3) }
#optimizer = {'name': "rmsprop", 'value': "rmsprop"}

conf = {}
#conf[1428] = {
#    "number_of_units": 200 # 0.876131201 instead of 0.733266945
#    }