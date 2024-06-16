
import version_bert.word2vec_version.utility.parameters as param
import json
import pandas as pd

word2idx = json.load(open(param.path_to_load_dict, "r"))
csv_reader = pd.read_csv(param.csv_file_path, low_memory=False)