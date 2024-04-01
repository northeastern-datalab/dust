import pandas as pd
import glob, sys, os
import json, torch, random
import numpy as np
sys.path.append("../")
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



def RunDiversityAlgorithms(S_dict, q_dict, algorithm, query_name, k, metric, normalize, lmda, eplot_folder_path, cplot_folder_path, embedding_type, max_metric, compute_metric):
    stats_df = pd.DataFrame(columns = ["algorithm", "embedding_type", "query_name", "|S|", "|q|", "k", "algorithm_distance_function", "evaluation_distance_function", "with_query_flag", "normalized", "max_div_score", "max-min_div_score", "avg_div_score", "time_taken_(s)"])
    diversified_tuples = {} #dictionary with algorithm name as key and the tuples as value
    if "gmc" in algorithm or "all" in algorithm:
        current_algorithm = "gmc"
        print(f"Using GMC method.")
        gmc_results, gmc_metrics, gmc_embedding_plot = div_utl.gmc(S_dict = copy.deepcopy(S_dict), q_dict = copy.deepcopy(q_dict), k = k, metric = metric, lmda=lmda, normalize=normalize, max_metric=max_metric, compute_metric = compute_metric)

        # write code to save gmc results in result_sets folder
        diversified_tuples[current_algorithm] = gmc_results
        if compute_metric == True:
            gmc_embedding_plot.title(f'{metric.capitalize()} distance PCA Embeddings of {query_name} result by GMC')
            plt_name = current_algorithm + "__" + query_name + "__" + metric + "__k-" + str(k) +".jpg"
            gmc_embedding_plot.savefig(eplot_folder_path + plt_name)

        for each in gmc_metrics:
            # each = {"metric": "l2", "with_query" : "yes", "max_score": l2_with_query_max_scores, "max-min_score": min(l2_with_query_min_scores), "avg_score": l2_with_query_avg_scores}
            append_list = [current_algorithm, embedding_type, query_name, len(S_dict), len(q_dict), k, metric, each['metric'], each["with_query"], normalize, each["max_score"], each["max-min_score"], each["avg_score"], each["time_taken"]]
            stats_df.loc[len(stats_df)] = append_list
        #stats_df_path = r"div_stats" + os.sep + benchmark_name + "__"+ current_algorithm + "_" + metric + ".csv"
        #stats_df.to_csv(stats_df_path)


    if "gne" in algorithm or "all" in algorithm:
        print(f"Using GNE method.")
        current_algorithm = "gne"
        gne_results, gne_metrics, gne_embedding_plot = div_utl.grasp(S_dict = copy.deepcopy(S_dict), q_dict=copy.deepcopy(q_dict), k = k, i_max=10, metric = metric, lmda= lmda, normalize=normalize, max_metric=max_metric, compute_metric = compute_metric)
        diversified_tuples[current_algorithm] = gne_results
        if compute_metric == True:
            gne_embedding_plot.title(f'{metric.capitalize()} distance PCA Embeddings of {query_name} result by GNE')
            plt_name = current_algorithm + "__" + query_name + "__" + metric + "__k-" + str(k) +".jpg"
            gne_embedding_plot.savefig(eplot_folder_path + plt_name)
        for each in gne_metrics:
            # each = {"metric": "l2", "with_query" : "yes", "max_score": l2_with_query_max_scores, "max-min_score": min(l2_with_query_min_scores), "avg_score": l2_with_query_avg_scores}
            append_list = [current_algorithm, embedding_type, query_name, len(S_dict), len(q_dict), k, metric, each['metric'], each["with_query"], normalize, each["max_score"], each["max-min_score"], each["avg_score"], each["time_taken"]]
            stats_df.loc[len(stats_df)] = append_list
        #stats_df_path = r"div_stats" + os.sep + benchmark_name + "__"+ current_algorithm + "_" + metric + ".csv"
        #stats_df.to_csv(stats_df_path)

    if "clt" in algorithm  or "all" in algorithm:
        print(f"Using clt method.")
        current_algorithm = "clt"
        clt_results, clt_metrics, clt_embedding_plot, clt_cluster_plot = div_utl.cluster_tuples(embedding_dict = copy.deepcopy(S_dict), q_dict = copy.deepcopy(q_dict), k = k, lmda = 0.7, method= "hierarchical", metric = metric, linkage="average", normalize=normalize, max_metric=max_metric, compute_metric = compute_metric)
        diversified_tuples[current_algorithm] = clt_results
        if compute_metric == True:
            clt_embedding_plot.title(f'{metric.capitalize()} distance PCA Embeddings of {query_name} result by CLT')
            plt_name = current_algorithm + "__" + query_name + "__" + metric + "__k-" + str(k) +".jpg"
            clt_embedding_plot.savefig(eplot_folder_path + plt_name)

            clt_cluster_plot.title(f'{metric.capitalize()} distance PCA Clusters of {query_name} result by CLT')
            plt_name = current_algorithm + "__" + query_name + "__" + metric + "__k-" + str(k) +".jpg"
            clt_cluster_plot.savefig(cplot_folder_path + plt_name)

        for each in clt_metrics:
            # each = {"metric": "l2", "with_query" : "yes", "max_score": l2_with_query_max_scores, "max-min_score": min(l2_with_query_min_scores), "avg_score": l2_with_query_avg_scores}
            append_list = [current_algorithm, embedding_type, query_name, len(S_dict), len(q_dict), k, metric, each['metric'], each["with_query"], normalize, each["max_score"], each["max-min_score"], each["avg_score"], each["time_taken"]]
            stats_df.loc[len(stats_df)] = append_list
        #stats_df_path = r"div_stats" + os.sep + benchmark_name + "__"+ current_algorithm + "_" + metric + ".csv"
        #stats_df.to_csv(stats_df_path)
    
    if "our_base" in algorithm  or "all" in algorithm:
        print(f"Using our_base method.")
        current_algorithm = "our_base"
        clt_results, clt_metrics, clt_embedding_plot, clt_cluster_plot = div_utl.cluster_tuples(embedding_dict = copy.deepcopy(S_dict), q_dict = copy.deepcopy(q_dict), k = min(len(S_dict), 2 * k), lmda = 0.7, method= "hierarchical", metric = metric, linkage="average", normalize=normalize, max_metric=max_metric, compute_metric = compute_metric)
        diversified_tuples[current_algorithm] = clt_results
        if compute_metric == True:
            clt_embedding_plot.title(f'{metric.capitalize()} distance PCA Embeddings of {query_name} result by DUST (base)')
            plt_name = current_algorithm + "__" + query_name + "__" + metric + "__k-" + str(k) +".jpg"
            clt_embedding_plot.savefig(eplot_folder_path + plt_name)

            clt_cluster_plot.title(f'{metric.capitalize()} distance PCA Clusters of {query_name} result by DUST (base)')
            plt_name = current_algorithm + "__" + query_name + "__" + metric + "__k-" + str(k) +".jpg"
            clt_cluster_plot.savefig(cplot_folder_path + plt_name)

        for each in clt_metrics:
            # each = {"metric": "l2", "with_query" : "yes", "max_score": l2_with_query_max_scores, "max-min_score": min(l2_with_query_min_scores), "avg_score": l2_with_query_avg_scores}
            append_list = [current_algorithm, embedding_type, query_name, len(S_dict), len(q_dict), k, metric, each['metric'], each["with_query"], normalize, each["max_score"], each["max-min_score"], each["avg_score"], each["time_taken"]]
            stats_df.loc[len(stats_df)] = append_list
        #stats_df_path = r"div_stats" + os.sep + benchmark_name + "__"+ current_algorithm + "_" + metric + ".csv"
        #stats_df.to_csv(stats_df_path)

    if "our" in algorithm  or "all" in algorithm:
        print(f"Using Our method.")
        current_algorithm = "our"
        our_results, our_metrics, our_embedding_plot, our_cluster_plot = div_utl.our_algorithm(embedding_dict = copy.deepcopy(S_dict), query_dict = copy.deepcopy(q_dict), k = k, method = "hierarchical", metric = metric, linkage="average", lmda = 0.7, strategy = "min", normalize=normalize, max_metric=max_metric, compute_metric = compute_metric)
        diversified_tuples[current_algorithm] = our_results
        if compute_metric == True:
            our_embedding_plot.title(f'{metric.capitalize()} distance PCA Embeddings of {query_name} result by Our algorithm')
            plt_name = current_algorithm + "__" + query_name + "__" + metric + "__k-" + str(k) +".jpg"
            our_embedding_plot.savefig(eplot_folder_path + plt_name)

            our_cluster_plot.title(f'{metric.capitalize()} distance PCA Clusters of {query_name} result by Our algorithm')
            plt_name = current_algorithm + "__" + query_name + "__" + metric + "__k-" + str(k) +".jpg"
            our_cluster_plot.savefig(cplot_folder_path + plt_name)

        for each in our_metrics:
            # each = {"metric": "l2", "with_query" : "yes", "max_score": l2_with_query_max_scores, "max-min_score": min(l2_with_query_min_scores), "avg_score": l2_with_query_avg_scores}
            append_list = [current_algorithm, embedding_type, query_name, len(S_dict), len(q_dict), k, metric, each['metric'], each["with_query"], normalize, each["max_score"], each["max-min_score"], each["avg_score"], each["time_taken"]]
            stats_df.loc[len(stats_df)] = append_list
    return diversified_tuples, stats_df
    #stats_df.to_csv(stats_df_path, index = False)
run_sample = "regular"  # {"sample", "efficiency_s", "efficiency_s", "efficiency_large", "regular"}
# query_name = r"sample_query.csv"
benchmark_name = r"ugen_benchmark" #will be the name of stat file
if run_sample == "efficiency_k" or run_sample == "efficiency_s":
    benchmark_name = r"efficiency_benchmark"
if run_sample == "efficiency_large":
    benchmark_name = r"efficiency_large_benchmark"
k = 30 #30
lmda = 0.7
# algorithm = {"all"} # gmc, gne, clt, our, all
s_dict_max = 2500
q_dict_max = 100
algorithm =  {"gmc", "clt", "our"} # {"all"} 
algorithm = {"our"}
metric = "cosine" # cosine, l1, l2
embedding_type = "dust"
eplot_folder_path = r"div_plots" + os.sep + "embedding_plots" + os.sep 
cplot_folder_path = r"div_plots" + os.sep + "cluster_plots" + os.sep 
result_folder_path = r"div_result_tables" + os.sep
algorithm_text = "_".join(algorithm)
stats_df_path = r"final_stats" + os.sep + benchmark_name + "__" + metric + "__" + embedding_type + "__" + algorithm_text + ".csv"
normalize = True
max_metric = False
compute_metric = True
full_dust = False
save_results = False
allowed_algorithms = {"all", "gmc", "gne", "clt", "our_base", "our"}
# div_result_path = r"div_result_tables" + os.sep + benchmark_name + os.sep + metric + os.sep + embedding_type + os.sep
div_result_path = os.path.join(r"div_result_tables", benchmark_name, metric, embedding_type)
# Create directory if it does not exist
if not os.path.exists(div_result_path):
    os.makedirs(div_result_path)

print("Selected algorithms:", algorithm)
if len(algorithm.intersection(allowed_algorithms)) == 0:
    print("Unsupported algorithm selected. Select one algorithm among: ", allowed_algorithms)
    sys.exit()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
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
    model_path = r'../out_model/tus_benchmark_corrected_roberta/checkpoints/best-checkpoint.pt'
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained('roberta-base')
    model = BertClassifier(model, num_labels = 2, hidden_size = 768, output_size = 768)
    model = DataParallel(model, device_ids=[0, 1, 2, 3])
    #print(model)   
    model.load_state_dict(torch.load(model_path)) # .to(device)
else:
    print("invalid embedding type")
    sys.exit()
# Load pre-trained Sentence-BERT model (replace 'embedding_type' with the name of the model you're using)
# model = SentenceTransformer('bert-base-uncased').to(device)
if run_sample != "efficiency_s" and run_sample != "efficiency_k" and run_sample != "efficiency_large":
    union_groundtruth_file_path = f"../groundtruth/{benchmark_name}_union_groundtruth.pickle"
    union_groundtruth = utl.loadDictionaryFromPickleFile(union_groundtruth_file_path)
union_datalake_folder_path = f"../data/{benchmark_name}/datalake/"
union_query_folder_path = f"../data/{benchmark_name}/query/"
all_stats_df = pd.DataFrame(columns = ["algorithm", "query_name", "|S|", "|q|", "k", "algorithm_distance_function", "evaluation_distance_function", "with_query_flag", "normalized", "max_div_score", "max-min_div_score", "avg_div_score", "time_taken_(s)"])

if run_sample == "sample":
    query_name = "sample_query.csv"
    benchmark_name = "sample_benchmark.csv"
    stats_df_path = r"div_stats" + os.sep + benchmark_name + "__" + metric + ".csv"
    dl_entities = utl.loadDictionaryFromPickleFile("./data/dl_entities.pickle")
    query_entities = utl.loadDictionaryFromPickleFile("./data/query_entities.pickle")
    S_dict =  utl.EmbedTuples(list(dl_entities.keys()), model, "sentence_bert", tokenizer, 1000)
    S_dict = dict(zip(list(dl_entities.keys()), S_dict))
    q_dict =  utl.EmbedTuples(list(query_entities.keys()), model, "sentence_bert", tokenizer, 1000)
    q_dict = dict(zip(list(query_entities.keys()), q_dict))
    diversified_tuples, current_stats = RunDiversityAlgorithms(S_dict, q_dict, algorithm, query_name, k, metric, normalize, lmda, eplot_folder_path, cplot_folder_path, embedding_type, max_metric, compute_metric = True)
    all_stats_df = pd.concat([all_stats_df, current_stats], axis = 0)
    all_stats_df.to_csv(stats_df_path, index = False)
elif run_sample == "efficiency_s":
    compute_metric = False
    print("\n=========Efficiency Experiment============\n")
    query_name = glob.glob(union_query_folder_path + "*.csv")[0].rsplit(os.sep, 1)[-1]
    benchmark_name = "efficiency_benchmark"
    print("Current Query: ", query_name)
    data_lake_tables = glob.glob(union_datalake_folder_path + "*.csv")
    # read the tables and collect their tuples as a list
    for dl_table in data_lake_tables:
        tuple_id = 0
        dl_tuple_dict = {}
        dl_table_name = dl_table.rsplit(os.sep, 1)[-1]
        print("Current data lake: ", dl_table_name)
        current_dl_table = utl.read_csv_file(dl_table)
        serialized_tuples = utl.SerializeTable(current_dl_table)
        for tup in serialized_tuples:
            dl_tuple_dict[tuple_id] = tup
            tuple_id += 1
        S_dict = utl.EmbedTuples(list(dl_tuple_dict.values()), model, embedding_type,tokenizer, 1000)
        S_dict = dict(zip(list(dl_tuple_dict.keys()), S_dict))
        print("Total data lake tuples:", len(dl_tuple_dict))
        query_tuple_dict = {}
        query_table = utl.read_csv_file(union_query_folder_path + query_name)
        serialized_tuples = utl.SerializeTable(query_table)
        for tup in serialized_tuples:
            query_tuple_dict[tuple_id] = tup
            tuple_id += 1
        q_dict = utl.EmbedTuples(list(query_tuple_dict.values()), model, embedding_type,tokenizer, 1000)
        q_dict = dict(zip(list(query_tuple_dict.keys()), q_dict))
        print("Total query tuples:", len(query_tuple_dict))
        diversified_tuples, current_stats = RunDiversityAlgorithms(S_dict, q_dict, algorithm, dl_table_name, k, metric, normalize, lmda, eplot_folder_path, cplot_folder_path, embedding_type, max_metric, compute_metric = compute_metric)
        all_stats_df = pd.concat([all_stats_df, current_stats], axis = 0)
        all_stats_df.to_csv(stats_df_path, index = False)
elif run_sample == "efficiency_k":
    compute_metric = False
    print("\n=========Efficiency Experiment============\n")
    query_name = glob.glob(union_query_folder_path + "*.csv")[0].rsplit(os.sep, 1)[-1]
    benchmark_name = "efficiency_benchmark"
    print("Current Query: ", query_name)
    data_lake_tables = glob.glob(union_datalake_folder_path + "5000.csv")
    # read the tables and collect their tuples as a list
    for k in range(50, 501, 50):
        tuple_id = 0
        dl_tuple_dict = {}
        dl_table = data_lake_tables[0]
        dl_table_name = dl_table.rsplit(os.sep, 1)[-1]
        print("Current data lake: ", dl_table_name)
        current_dl_table = utl.read_csv_file(dl_table)
        serialized_tuples = utl.SerializeTable(current_dl_table)
        for tup in serialized_tuples:
            dl_tuple_dict[tuple_id] = tup
            tuple_id += 1
        S_dict = utl.EmbedTuples(list(dl_tuple_dict.values()), model, embedding_type,tokenizer, 1000)
        S_dict = dict(zip(list(dl_tuple_dict.keys()), S_dict))
        print("Total data lake tuples:", len(dl_tuple_dict))
        query_tuple_dict = {}
        query_table = utl.read_csv_file(union_query_folder_path + query_name)
        serialized_tuples = utl.SerializeTable(query_table)
        for tup in serialized_tuples:
            query_tuple_dict[tuple_id] = tup
            tuple_id += 1
        q_dict = utl.EmbedTuples(list(query_tuple_dict.values()), model, embedding_type,tokenizer, 1000)
        q_dict = dict(zip(list(query_tuple_dict.keys()), q_dict))
        print("Total query tuples:", len(query_tuple_dict))
        diversified_tuples, current_stats = RunDiversityAlgorithms(S_dict, q_dict, algorithm, dl_table_name, k, metric, normalize, lmda, eplot_folder_path, cplot_folder_path, embedding_type, max_metric, compute_metric = compute_metric)
        all_stats_df = pd.concat([all_stats_df, current_stats], axis = 0)
        all_stats_df.to_csv(stats_df_path, index = False)
elif run_sample == "efficiency_large":
    compute_metric = False
    algorithm = {"our"}
    print("\n=========Efficiency Experiment============\n")
    query_name = glob.glob(union_query_folder_path + "*.csv")[0].rsplit(os.sep, 1)[-1]
    benchmark_name = "efficiency_large_benchmark"
    print("Current Query: ", query_name)
    data_lake_tables = glob.glob(union_datalake_folder_path + "*.csv")
    # read the tables and collect their tuples as a list
    for dl_table in data_lake_tables:
        tuple_id = 0
        dl_tuple_dict = {}
        dl_table_name = dl_table.rsplit(os.sep, 1)[-1]
        print("Current data lake: ", dl_table_name)
        current_dl_table = utl.read_csv_file(dl_table)
        serialized_tuples = utl.SerializeTable(current_dl_table)
        for tup in serialized_tuples:
            dl_tuple_dict[tuple_id] = tup
            tuple_id += 1
        S_dict = utl.EmbedTuples(list(dl_tuple_dict.values()), model, embedding_type,tokenizer, 1000)
        S_dict = dict(zip(list(dl_tuple_dict.keys()), S_dict))
        print("Total data lake tuples:", len(dl_tuple_dict))
        query_tuple_dict = {}
        query_table = utl.read_csv_file(union_query_folder_path + query_name)
        serialized_tuples = utl.SerializeTable(query_table)
        for tup in serialized_tuples:
            query_tuple_dict[tuple_id] = tup
            tuple_id += 1
        q_dict = utl.EmbedTuples(list(query_tuple_dict.values()), model, embedding_type,tokenizer, 1000)
        q_dict = dict(zip(list(query_tuple_dict.keys()), q_dict))
        print("Total query tuples:", len(query_tuple_dict))
        diversified_tuples, current_stats = RunDiversityAlgorithms(S_dict, q_dict, algorithm, dl_table_name, k, metric, normalize, lmda, eplot_folder_path, cplot_folder_path, embedding_type, max_metric, compute_metric = compute_metric)
        all_stats_df = pd.concat([all_stats_df, current_stats], axis = 0)
        all_stats_df.to_csv(stats_df_path, index = False)
else:
    for query_name in union_groundtruth:
        try:
            print("\n========================================\n")
            print("Current Query: ", query_name)
            if not os.path.exists(union_query_folder_path + query_name):
                continue
            unionable_tables = union_groundtruth[query_name]
            # if benchmark_name == "tus_benchmark":
            #     unionable_table_path = [union_datalake_folder_path + os.sep + tab for tab in unionable_tables if tab != query_name]
            #     unionable_table_path = [path for path in unionable_table_path if os.path.exists(path)]
            #     random.seed(random_seed) # we use random seed =  42 in all experiments.
            #     unionable_table_path = random.sample(unionable_table_path, min(10, len(unionable_table_path))) 
            #     unionable_tables = [tabl.rsplit(os.sep, 1)[-1] for tabl in unionable_table_path] 
            print("Total unionable tables: ", len(unionable_tables))
            query_table = utl.read_csv_file(union_query_folder_path + query_name)
            columns_in_query = set(query_table.columns.astype(str))
            # read the tables and collect their tuples as a list
            tuple_id = 0
            dl_tuple_dict = {}
            for dl_table in unionable_tables:
                current_dl_table = utl.read_csv_file(union_datalake_folder_path + dl_table)
                current_dl_table.columns = current_dl_table.columns.astype(str)
                if full_dust == True: 
                    #alignment in dataset is already done in previous phase, we only need to drop the columns not in query.
                    columns_to_drop = set(current_dl_table.columns.astype(str)) - columns_in_query
                    current_dl_table = current_dl_table.drop(columns=columns_to_drop)
                serialized_tuples = utl.SerializeTable(current_dl_table)
                for tup in serialized_tuples:
                    dl_tuple_dict[tuple_id] = tup
                    tuple_id += 1
            if len(dl_tuple_dict) > s_dict_max:
                random.seed(random_seed)
                sampled_keys = random.sample(dl_tuple_dict.keys(), s_dict_max)
                sampled_dict = {key: dl_tuple_dict[key] for key in sampled_keys}
                dl_tuple_dict = sampled_dict

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
                sampled_keys = random.sample(query_tuple_dict.keys(), q_dict_max)
                sampled_dict = {key: query_tuple_dict[key] for key in sampled_keys}
                query_tuple_dict = sampled_dict
            q_dict = utl.EmbedTuples(list(query_tuple_dict.values()), model, embedding_type,tokenizer, 1000)
            q_dict = dict(zip(list(query_tuple_dict.keys()), q_dict))
            print("Total query tuples:", len(query_tuple_dict))
            if len(q_dict) < 3:
                print(f"Query table: {query_name} has only {len(q_dict)} rows. So, ignoring this table.")
                continue
            diversified_tuples, current_stats = RunDiversityAlgorithms(S_dict, q_dict, algorithm, query_name, k, metric, normalize, lmda, eplot_folder_path, cplot_folder_path, embedding_type, max_metric, compute_metric= True)
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