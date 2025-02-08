# %%
import glob, random
import json, sys, os
from pathlib import Path
import div_utilities as div_utl
import utilities as utl
import torch
from transformers import BertTokenizer, BertModel, RobertaTokenizerFast, RobertaModel
from model_classes import BertClassifierPretrained, BertClassifier
from glove_embeddings import GloveTransformer
import fasttext_embeddings as ft
from torch.nn.parallel import DataParallel
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine, euclidean


random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
embedding_type = "dust"
# device = "cpu"
print("Model type: ", embedding_type)
if embedding_type == "bert":
    model = BertModel.from_pretrained('bert-base-uncased') 
    model = BertClassifierPretrained(model).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vec_length = 768
elif embedding_type == "roberta":
    model = RobertaModel.from_pretrained("roberta-base")
    model = BertClassifierPretrained(model).to(device)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    vec_length = 768
elif embedding_type == "sentence_bert":
    model = SentenceTransformer('bert-base-uncased').to(device) #case insensitive model. BOSTON and boston have the same embedding.
    tokenizer = ""
    vec_length = 768
elif embedding_type == "glove":
    model = GloveTransformer()
    tokenizer = ""
    vec_length = 300
elif embedding_type == "fasttext":
    model = ft.get_embedding_model()
    tokenizer = ""
    vec_length = 300
elif embedding_type == "dust":
    model_path = r'../out_model/tus_finetune_roberta/checkpoints/best-checkpoint.pt'
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained('roberta-base')
    model = BertClassifier(model, num_labels = 2, hidden_size = 768, output_size = 768)
    model = DataParallel(model, device_ids=[0, 1, 2, 3])
    #print(model)   
    model.load_state_dict(torch.load(model_path)) # .to(device)
else:
    print("invalid embedding type")
    sys.exit()


# %%
csv_dir = "/home/khatiwada/dust/data/tables/"
sd_stats = "/home/khatiwada/dust/diversity_algorithms/sd_stats/"
all_tuple_dict = {} # (table, row_id) -> np.array(756)
per_table_stats = {}
per_row_distance_from_mean = {}
table_id = 0
all_tables = glob.glob(f"{csv_dir}*") #Path(csv_dir).glob('*')
print("Total tables: ", len(all_tables))
for fpath in tqdm(all_tables, desc = "Embedding"):
    with open(fpath, errors = 'backslashreplace') as f:
        # print(f"reading table: {fpath}")
        try:
            current_tuple_dict = {}
            table = pd.read_csv(f, nrows=5000, header=None, on_bad_lines="skip", keep_default_na=False, dtype=str).replace('', 'nan', regex=True).replace("\n", '', regex=True).replace(",", "", regex=True).replace("'", "", regex=True).replace('"', '', regex=True)
            serialized_tuples = utl.SerializeTable(table)
            for idx, tup in enumerate(serialized_tuples):
                current_tuple_dict[(fpath.rsplit(os.sep, 1)[-1], idx)] = tup
            S_dict = utl.EmbedTuples(list(current_tuple_dict.values()), model, embedding_type,tokenizer, 1000)
            S_dict = dict(zip(list(current_tuple_dict.keys()), S_dict))
            for table_rowid in S_dict:
                all_tuple_dict[table_rowid] = S_dict[table_rowid]
            current_table_array = np.array(list(S_dict.values()))
            current_table_mean = np.mean(current_table_array, axis=0)
            # current_table_std_devs = np.std(current_table_array, axis=1)
            cosine_distances = [cosine(embedding, current_table_mean) for embedding in list(S_dict.values())]
            euclidean_distances = [euclidean(embedding, current_table_mean) for embedding in list(S_dict.values())]
            std_dev_of_cosine_distances = np.std(cosine_distances)
            std_dev_of_euclidean_distances = np.std(euclidean_distances)
            per_table_stats[fpath.rsplit(os.sep, 1)[-1]] = {"mean_embedding": current_table_mean,
                                                             "cosine_sd": std_dev_of_cosine_distances,
                                                               "euclidean_sd": std_dev_of_euclidean_distances}
            for idx, table_rowid in enumerate(list(S_dict.keys())):
                per_row_distance_from_mean[table_rowid] = {"cosine": cosine_distances[idx],
                                                            "euclidean": euclidean_distances[idx]}
            table_id += 1
            if table_id % 50 == 0:
                utl.saveDictionaryAsPickleFile(per_table_stats, sd_stats + "per_table_stats.pickle")
                utl.saveDictionaryAsPickleFile(per_row_distance_from_mean, sd_stats + "per_row_distance_from_mean.pickle")
                utl.saveDictionaryAsPickleFile(all_tuple_dict, sd_stats + "all_tuple_dict.pickle")
            if len(per_table_stats) >= 1000:
                break
        
        except Exception as e:
             print(e)
             print("Bad table:", fpath.rsplit(os.sep, 1)[-1])

# %%
utl.saveDictionaryAsPickleFile(per_table_stats, sd_stats + "per_table_stats.pickle")
utl.saveDictionaryAsPickleFile(per_row_distance_from_mean, sd_stats + "per_row_distance_from_mean.pickle")
utl.saveDictionaryAsPickleFile(all_tuple_dict, sd_stats + "all_tuple_dict.pickle")


