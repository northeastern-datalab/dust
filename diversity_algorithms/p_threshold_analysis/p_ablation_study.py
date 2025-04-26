import pandas as pd
import glob, sys, os
import json, torch, random
import numpy as np
sys.path.append("../")
sys.path.append("../../")

import utilities as utl
from sentence_transformers import SentenceTransformer
from pympler import asizeof
from sklearn.cluster import KMeans, AgglomerativeClustering
from bkmeans import BKMeans
from sklearn.metrics import pairwise_distances
import div_utilities as div_utl
import copy
from transformers import BertTokenizer, BertModel, RobertaTokenizerFast, RobertaModel
from model_classes import BertClassifierPretrained, BertClassifier
from glove_embeddings import GloveTransformer
import fasttext_embeddings as ft
from torch.nn.parallel import DataParallel
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)



def RunDiversityAlgorithms(S_dict, q_dict, algorithm, query_name, k, metric, normalize, lmda, eplot_folder_path, cplot_folder_path, embedding_type, max_metric, compute_metric, p_values = [2]):
    stats_df = pd.DataFrame(columns = ["algorithm", "embedding_type", "query_name", "|S|", "|q|", "k", "algorithm_distance_function", "evaluation_distance_function", "with_query_flag", "normalized", "max_div_score", "max-min_div_score", "avg_div_score", "time_taken_(s)"])
    diversified_tuples = {} #dictionary with algorithm name as key and the tuples as value
    for p in p_values:
        current_algorithm = "dust_base-p=" + str(p)
        print(f"Using {current_algorithm} method.")
        our_results, our_metrics, our_embedding_plot, our_cluster_plot = div_utl.our_algorithm(embedding_dict = copy.deepcopy(S_dict), query_dict = copy.deepcopy(q_dict), k = k, method = "hierarchical", metric = metric, linkage="average", lmda = 0.7, strategy = "min", normalize=normalize, max_metric=max_metric, compute_metric = compute_metric, s_dict_max= s_dict_max, p = p)
        diversified_tuples[current_algorithm] = our_results
        if compute_metric == True:
            our_embedding_plot.title(f'{metric.capitalize()} distance PCA Embeddings of {query_name} result by  DUST p = {str(p)}')
            plt_name = current_algorithm + "__" + query_name + "__" + metric + "__k-" + str(k) +".jpg"
            our_embedding_plot.savefig(eplot_folder_path + plt_name)

            our_cluster_plot.title(f'{metric.capitalize()} distance PCA Clusters of {query_name} result by  DUST p = {str(p)}')
            plt_name = current_algorithm + "__" + query_name + "__" + metric + "__k-" + str(k) +".jpg"
            our_cluster_plot.savefig(cplot_folder_path + plt_name)

        for each in our_metrics:
            # each = {"metric": "l2", "with_query" : "yes", "max_score": l2_with_query_max_scores, "max-min_score": min(l2_with_query_min_scores), "avg_score": l2_with_query_avg_scores}
            append_list = [current_algorithm, embedding_type, query_name, len(S_dict), len(q_dict), k, metric, each['metric'], each["with_query"], normalize, each["max_score"], each["max-min_score"], each["avg_score"], each["time_taken"]]
            stats_df.loc[len(stats_df)] = append_list
        #stats_df_path = r"div_stats" + os.sep + benchmark_name + "__"+ current_algorithm + "_" + metric + ".csv"
        #stats_df.to_csv(stats_df_path)

    return diversified_tuples, stats_df
    #stats_df.to_csv(stats_df_path, index = False)

k = 30 #30 or 100
lmda = 0.7
# algorithm = {"all"} # gmc, gne, clt, our, all
s_dict_max = 2500
q_dict_max = 100
benchmark_name = r"ugen_benchmark" #will be the name of stat file
algorithm = {"our_base"} 
p_values = [1, 2, 3, 4, 5]
metric = "cosine" # cosine, l1, l2
embedding_type = "dust"
eplot_folder_path = r"../div_plots" + os.sep + "embedding_plots" + os.sep 
cplot_folder_path = r"../div_plots" + os.sep + "cluster_plots" + os.sep 
result_folder_path = r"../div_result_tables" + os.sep
algorithm_text = "_".join(algorithm)
# algorithm_text += "unpruned"
stats_df_path = r"../final_stats" + os.sep + benchmark_name + "__" + metric + "__" + embedding_type + "__" + algorithm_text + ".csv"
normalize = True
max_metric = False
compute_metric = True
save_results = True
# div_result_path = r"div_result_tables" + os.sep + benchmark_name + os.sep + metric + os.sep + embedding_type + os.sep
div_result_path = os.path.join(r"../div_result_tables", benchmark_name, metric, embedding_type)
# Create directory if it does not exist
if not os.path.exists(div_result_path):
    os.makedirs(div_result_path)

print("Selected algorithms:", algorithm)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
# device = "cpu"
print("Model type: ", embedding_type)

model_path = r'../../out_model/tus_finetune_roberta/checkpoints/best-checkpoint.pt'
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained('roberta-base')
model = BertClassifier(model, num_labels = 2, hidden_size = 768, output_size = 768)
model = DataParallel(model, device_ids=[0, 1, 2, 3])
#print(model)   
model.load_state_dict(torch.load(model_path)) # .to(device)
# model = SentenceTransformer('bert-base-uncased').to(device)
union_groundtruth_file_path = f"../../groundtruth/{benchmark_name}_union_groundtruth.pickle"
union_groundtruth = utl.loadDictionaryFromPickleFile(union_groundtruth_file_path)
union_datalake_folder_path = f"../../data/{benchmark_name}/datalake/"
union_query_folder_path = f"../../data/{benchmark_name}/query/"
all_stats_df = pd.DataFrame(columns = ["algorithm", "query_name", "|S|", "|q|", "k", "algorithm_distance_function", "evaluation_distance_function", "with_query_flag", "normalized", "max_div_score", "max-min_div_score", "avg_div_score", "time_taken_(s)"])

for query_name in union_groundtruth:
    try:
        print("\n========================================\n")
        print("Current Query: ", query_name)
        if not os.path.exists(union_query_folder_path + query_name):
            continue
        unionable_tables = union_groundtruth[query_name]
        print("Total unionable tables: ", len(unionable_tables))
        query_table = utl.read_csv_file(union_query_folder_path + query_name)
        columns_in_query = set(query_table.columns.astype(str))
        # read the tables and collect their tuples as a list
        tuple_id = 0
        dl_tuple_dict = {}
        for dl_table in unionable_tables:
            current_dl_table = utl.read_csv_file(union_datalake_folder_path + dl_table)
            current_dl_table.columns = current_dl_table.columns.astype(str)
            serialized_tuples = utl.SerializeTable(current_dl_table)
            for tup in serialized_tuples:
                dl_tuple_dict[tuple_id] = tup
                tuple_id += 1
        if len(dl_tuple_dict) > s_dict_max: #and "our" not in algorithm: # We select random S tuples for the baselines as they do not scale for larger S.
            random.seed(random_seed)
            try:
                sampled_keys = random.sample(dl_tuple_dict.keys(), s_dict_max)
                sampled_dict = {key: dl_tuple_dict[key] for key in sampled_keys}
                dl_tuple_dict = sampled_dict
            except Exception as e:
                print("sampling did not work. less than: ", e)
        S_dict = utl.EmbedTuples(list(dl_tuple_dict.values()), model, embedding_type,tokenizer, 1000)
        S_dict = dict(zip(list(dl_tuple_dict.keys()), S_dict))
        print("Total data lake tuples:", len(dl_tuple_dict))
        # print("S_dict keys: ", S_dict.keys())
        # break
        if k > len(S_dict): 
            print(f"Data lake has {len(S_dict)} tuples but k = {k}. So, ignoring this table.")
            continue
        query_tuple_dict = {}
        serialized_tuples = utl.SerializeTable(query_table)
        for tup in serialized_tuples:
            query_tuple_dict[tuple_id] = tup
            tuple_id += 1
        if len(query_tuple_dict) > q_dict_max:
            random.seed(random_seed)
            sampled_keys = random.sample(list(query_tuple_dict.keys()), q_dict_max)
            sampled_dict = {key: query_tuple_dict[key] for key in sampled_keys}
            query_tuple_dict = sampled_dict
        q_dict = utl.EmbedTuples(list(query_tuple_dict.values()), model, embedding_type,tokenizer, 1000)
        q_dict = dict(zip(list(query_tuple_dict.keys()), q_dict))
        print("Total query tuples:", len(query_tuple_dict))
        if len(q_dict) < 3:
            print(f"Query table: {query_name} has only {len(q_dict)} rows. So, ignoring this table.")
            continue
        diversified_tuples, current_stats = RunDiversityAlgorithms(S_dict, q_dict, algorithm, query_name, k, metric, normalize, lmda, eplot_folder_path, cplot_folder_path, embedding_type, max_metric, compute_metric= True, p_values=p_values)
        all_stats_df = pd.concat([all_stats_df, current_stats], axis = 0)
        all_stats_df.to_csv(stats_df_path, index = False)
        if save_results == True:
            for technique in diversified_tuples:
                f_path = os.path.join(div_result_path, technique)
                if not os.path.exists(f_path):
                    os.makedirs(f_path)
                c_diversified_tuples = diversified_tuples[technique]
                # print("Dl dict:", dl_tuple_dict)
                current_div_results_path = f_path + os.sep + query_name.rsplit(".",1)[0] + ".txt"
                with open(current_div_results_path, "w") as f:
                    for div_tuple in c_diversified_tuples:
                        f.write(dl_tuple_dict[int(div_tuple)] + "\n")
    except Exception as e:
        print(e)

# todo: change diversified tuple id to the original tuples