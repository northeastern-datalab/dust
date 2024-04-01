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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

def debug_dict(dictionary, k, name = "Dictionary"):
    k = min(k, len(dictionary))
    random_sim_keys = random.sample(list(dictionary.keys()), k)
    print("Dict name:", name)
    for key in random_sim_keys:
        print(f"Key: {key} ; value : {dictionary[key]}")

# Function to recursively calculate the memory usage of an object and its contents
def get_object_memory_usage(obj):
    # Use pympler's asizeof to estimate memory usage
    # return 
    total_memory_usage = asizeof.asizeof(obj)
    print(f"Total memory usage of the dictionary: {total_memory_usage/ (1024 * 1024)} MB")

#generate dummy data.
def generate_sbert_dummy_embeddings(batch_size = 1000, total_embeddings = 5000, sentence_list = []):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    # Load pre-trained Sentence-BERT model (replace 'model_name' with the name of the model you're using)
    model = SentenceTransformer('bert-base-uncased').to(device)

    # Initialize an empty dictionary
    embedding_dict = {}

    sentences_batch = []

    if len(sentence_list) > 0:
        # Iterate from ID 1 to 1000
        for i in range(0, len(sentence_list)):
            # Generate random sentences or use your own logic to determine which sentences to assign
            sentence = sentence_list[i]  # For example, Sentence 1, Sentence 2, ...
            sentences_batch.append(sentence)
            
            # If the batch size is reached or it's the last sentence, embed the batch
            if len(sentences_batch) == batch_size or i == len(sentence_list) - 1:
                # Encode the batch of sentences to get Sentence-BERT embeddings
                sentence_embeddings = model.encode(sentences_batch, convert_to_tensor=True)  # Tensor of shape (batch_size, 768)

                # Convert the tensor to a list of NumPy arrays
                embeddings_list = sentence_embeddings.cpu().numpy()

                # Add the entries to the dictionary with IDs as the keys and embeddings as the values
                for j, embedding in enumerate(embeddings_list):
                    embedding_dict[sentences_batch[j]] = embedding

                # Clear the batch for the next set of sentences
                sentences_batch = []
                if i % 5000 == 0:
                    print(f"Done upto {i} items")
        return embedding_dict
    else:
        # Iterate from ID 1 to 1000
        for i in range(1, total_embeddings + 1):
            # Generate random sentences or use your own logic to determine which sentences to assign
            sentence = f'Sentence {i}'  # For example, Sentence 1, Sentence 2, ...
            sentences_batch.append(sentence)
            
            # If the batch size is reached or it's the last sentence, embed the batch
            if len(sentences_batch) == batch_size or i == 1000:
                # Encode the batch of sentences to get Sentence-BERT embeddings
                sentence_embeddings = model.encode(sentences_batch, convert_to_tensor=True)  # Tensor of shape (batch_size, 768)

                # Convert the tensor to a list of NumPy arrays
                embeddings_list = sentence_embeddings.cpu().numpy()

                # Add the entries to the dictionary with IDs as the keys and embeddings as the values
                for j, embedding in enumerate(embeddings_list):
                    embedding_dict[i - batch_size + j + 1] = embedding

                # Clear the batch for the next set of sentences
                sentences_batch = []
                if i % 5000 == 0:
                    print(f"Done upto {i} items")
        get_object_memory_usage(embedding_dict)

        q = model.encode(["random query embedding"], convert_to_tensor=True)
        q = q[0].cpu().numpy()
        return embedding_dict, q

# Code for baseline algorithms start here.

def d_sim(s_dict : dict, q_embedding: np.ndarray, metric = "cosine", normalize = False) -> dict:
    sim_dict = dict() # key: s_dict key i.e. s_id; value : similarity score
    for current_s, current_s_embedding in s_dict.items():
        if metric == "l1":
            if normalize == True:
                max_possible_l1 = 2 * len(current_s_embedding)
            else:
                max_possible_l1 = 1
            current_sim = np.linalg.norm(current_s_embedding - q_embedding, ord = 1) / max_possible_l1
        elif metric == "l2":
            if normalize == True:
                max_possible_l2 = np.sqrt(2 * len(current_s_embedding))
            else:
                max_possible_l2 = 1
            current_sim = np.linalg.norm(current_s_embedding - q_embedding, ord = 2) / max_possible_l2
        else: # cosine
            if normalize == True:
                current_sim = 1 - ((utl.CosineSimilarity(current_s_embedding, q_embedding) + 1 ) / 2)
            else:
                current_sim = 1 - utl.CosineSimilarity(current_s_embedding, q_embedding)
        sim_dict[current_s] = current_sim
    return sim_dict

def d_div(s_dict : dict, metric = "cosine", normalize = False) -> dict:
    div_dict = dict() # key: s_dict key i.e. s_id; value : similarity score
    for current_s1 in s_dict:
        for current_s2 in s_dict:
            if metric == "l1":
                max_possible_l1 = 2 * len(s_dict[current_s1])
                if normalize == True:
                    current_div = 1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 1) / max_possible_l1)
                else:
                    current_div = max_possible_l1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 1) / 1)                
            elif metric == "l2":
                max_possible_l2 = np.sqrt(2 * len(s_dict[current_s1]))
                if normalize == True:
                    current_div = 1 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 2) / max_possible_l2)
                else:
                    current_div = max_possible_l2 - (np.linalg.norm(s_dict[current_s1] - s_dict[current_s2], ord = 2) / 1)
            else: #cosine
                if normalize == True:
                    current_div = (utl.CosineSimilarity(s_dict[current_s1], s_dict[current_s2]) + 1) / 2 #normalized score between 0 and 1
                else:
                    current_div = utl.CosineSimilarity(s_dict[current_s1], s_dict[current_s2])
            div_dict[(current_s1, current_s2)] = current_div
    return div_dict

def CountClasses(metoids, entity_dict):
    count_classes = {}
    for each in metoids:
        class_name = entity_dict[each]
        if class_name in count_classes:
            count_classes[class_name] += 1
        else:
            count_classes[class_name] = 1
    return count_classes

def min_div_score(s_dict: dict, metric = "cosine", normalize = False) -> float:
    if len(s_dict) == 0:
        return [0]
    min_scores = [] # all possible distances
    for current_s1, current_s1_embedding in s_dict.items():
        for current_s2, current_s2_embedding in s_dict.items():
            if current_s1 != current_s2:
                if metric == "l1":
                    if normalize == True:
                        max_possible_l1 = 2 * len(current_s1_embedding)
                    else:
                        max_possible_l1 = 1
                    current_sim = np.linalg.norm(current_s1_embedding - current_s2_embedding, ord = 1) / max_possible_l1  # utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                elif metric == "l2":
                    if normalize == True:
                        max_possible_l2 = np.sqrt(2 * len(current_s1_embedding))
                    else:
                        max_possible_l2 = 1
                    current_sim = np.linalg.norm(current_s1_embedding - current_s2_embedding, ord = 2) / max_possible_l2 # utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                else: # metric = cosine
                    if normalize == True:
                        current_sim = 1 - ((utl.CosineSimilarity(current_s1_embedding, current_s2_embedding) + 1 ) / 2)
                    else:
                        current_sim = 1 - utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                min_scores.append(current_sim)
    return min_scores

def min_mix_div_score(s_dict: dict, q_set:set, metric = "cosine", normalize = False) -> float:
    if len(s_dict) == 0:
        return [0]
    min_scores = [] # all possible distances
    for current_s1, current_s1_embedding in s_dict.items():
        for current_s2, current_s2_embedding in s_dict.items():
            if current_s1 in q_set and current_s2 in q_set:
                continue
            if current_s1 != current_s2:
                if metric == "l1":
                    if normalize == True:
                        max_possible_l1 = 2 * len(current_s1_embedding)
                    else:
                        max_possible_l1 = 1
                    current_sim = np.linalg.norm(current_s1_embedding - current_s2_embedding, ord = 1) / max_possible_l1  # utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                elif metric == "l2":
                    if normalize == True:
                        max_possible_l2 = np.sqrt(2 * len(current_s1_embedding))
                    else:
                        max_possible_l2 = 1
                    current_sim = np.linalg.norm(current_s1_embedding - current_s2_embedding, ord = 2) / max_possible_l2 # utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                else: # metric = cosine
                    if normalize == True:
                        current_sim = 1 - ((utl.CosineSimilarity(current_s1_embedding, current_s2_embedding) + 1 ) / 2)
                    else:
                        current_sim = 1 - utl.CosineSimilarity(current_s1_embedding, current_s2_embedding)
                min_scores.append(current_sim)
    return min_scores

def mmc_compute_div_sum(s_i: str, div_dict: dict, R_p: set) -> float:
    total_score = 0
    for s_j in R_p:
        if (s_i , s_j) in div_dict:
            total_score += div_dict[(s_i, s_j)]
        else:
            total_score += div_dict[(s_j, s_i)]
    return total_score

def mmc_compute_div_large(s_i: str, div_dict: dict, remaining_s_set : set, max_l : int) -> float: #max_l should be used as: < max_l ; not <= max_l
    div_l_list = [] # we will use the first l values after sorting this.
    for s_j in remaining_s_set: # the items that are not inserted in R_p yet
        if (s_i, s_j) in div_dict:
            div_l_list.append(div_dict[(s_i, s_j)])
        else:
            div_l_list.append(div_dict[(s_j, s_i)])
    # print("Div l: ", div_l_list)
    # print("MAX L", max_l - 1)
    div_l_list = sorted(div_l_list, reverse=True)[:max_l - 1]
    return div_l_list

def gne_compute_div_large_key(s_i: str, div_dict: dict, s_set : set, max_l : int) -> list: #max_l should be used as: < max_l ; not <= max_l
    div_l_dict = dict() # we will use the first l values after sorting this.
    div_val = 0
    for s_j in s_set: # the items that are not inserted in R_p yet
        if (s_i, s_j) in div_dict:
            div_val = div_dict[(s_i, s_j)]
        else:
            div_val = div_dict[(s_j, s_i)]
        div_l_dict[s_j] = div_val
    # print("Div l: ", div_l_list)
    # print("MAX L", max_l - 1)
    div_l_list = sorted(div_l_dict.items(), key=lambda item: item[1])

    div_l_keys = [item[0] for item in div_l_list][:max_l - 1]
    return div_l_keys

def f_prime(s_set: set, lmda : float, k: int, sim_dict: dict, div_dict: dict) -> float:
    if len(s_set) == 0:
        return 0
    total_sim_score = 0
    total_div_score = 0
    for s_i in s_set:
        total_sim_score += sim_dict[s_i]
    for s_i in s_set:
        for s_j in s_set:
            if (s_i, s_j) in div_dict:
                total_div_score += div_dict[(s_i, s_j)]
            else:
                total_div_score += div_dict[(s_j, s_i)]
    total_sim_score *= (k-1) * (1 - lmda)
    total_div_score *= 2 * lmda
    return (total_sim_score + total_div_score)

def visualize_embeddings(datalake, query, show_plot = False):
    # Combine the embeddings
    embeddings = np.array(list(datalake.values()) + list(query.values()))

    # Apply PCA to reduce the dimensionality to 2 for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(embeddings)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))

    # Scatter plot for items from datalake (red color)
    plt.scatter(pca_result[:len(datalake), 0], pca_result[:len(datalake), 1], c='red', label='data lake', alpha=0.7)

    # Scatter plot for items from query (blue color)
    plt.scatter(pca_result[len(datalake):, 0], pca_result[len(datalake):, 1], c='blue', label='query', alpha=0.7)

    # Set plot labels and legend
    # plt.title('PCA Visualization of Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    # Show the plot
    if show_plot == True:
        plt.show()
    return plt

def visualize_clusters(embedding_dict, cluster_assignments, cluster_metoids, k, cluster_centroids = {}, show_plot = False):
    embeddings = np.array(list(embedding_dict.values()))
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    # Visualize clusters in the reduced space
    plt.figure(figsize=(8, 6))
    for i in range(k):
        cluster_items_indices = np.where(cluster_assignments == i)[0]
        cluster_items_pca = embeddings_pca[cluster_items_indices]
        plt.scatter(cluster_items_pca[:, 0], cluster_items_pca[:, 1], label=f'Cluster {i}')
    # Plot metoids
    metoids_embeddings = list(embedding_dict[key] for key in cluster_metoids.values())
    metoids_pca = pca.fit_transform(metoids_embeddings)
    for i, metoid_pca in enumerate(metoids_pca):
        plt.scatter(metoid_pca[0], metoid_pca[1], marker='X', color='black', s=100, label=f'Metoid {i}')
    # if len(cluster_centroids) > 0:
    #     centroid_embeddings = list(cluster_centroids.values())
    #     centroids_pca = pca.fit_transform(centroid_embeddings)
    #     for i, centroid_pca in enumerate(centroids_pca):
    #         plt.scatter(centroid_pca[0], centroid_pca[1], marker='o', color='red', s=100, label=f'Centroid {i}')
    #plt.title('Clusters Visualization using PCA')
    #plt.legend()
    if show_plot == True:
        plt.show()
    return plt

def compute_metrics(result, dl_embeddings:dict, query_embeddings:dict, lmda:float, k:float, print_results = False, normalize = False, metric = "", max_metric = True):
    computed_metrics = [] # list of dictionaries, each dict is a row in the evaluation dataframe. 
    q = np.mean(list(query_embeddings.values()), axis=0)
    ranking_without_query = {}
    for key in result:
        ranking_without_query[key] = dl_embeddings[key]
    embedding_plot = visualize_embeddings(ranking_without_query, query_embeddings, show_plot=print_results)
    final_ranking_with_query = ranking_without_query.copy()
    for key in query_embeddings:
        final_ranking_with_query[key] = query_embeddings[key]
        dl_embeddings[key] = query_embeddings[key] # we do not need separate data lake embeddings anymore, so merging with query.
    
    R_without_query = set(result)
    R_with_query = set(result).union(set(query_embeddings.keys()))
    if max_metric == True:
        if metric == "" or metric == "cosine":
            # Evaluating max diversity using cosine distance:
            sim_dict = d_sim(dl_embeddings, q, metric="cosine", normalize=normalize) # index the similarity between each item in S and q for once so that we can re-use them.
            div_dict = d_div(dl_embeddings, metric = "cosine", normalize=normalize) # index the diversity between each pair of items in S so that we can re-use them.
            cosine_with_query_max_scores = f_prime(R_with_query, lmda, k, sim_dict, div_dict)
            cosine_wo_query_max_scores =  f_prime(R_without_query, lmda, k, sim_dict, div_dict)
            
            if print_results == True:
                print("Evaluating max diversity using cosine distance:")
                print("max score with query: ", cosine_with_query_max_scores)
                print("max score without query: ", cosine_wo_query_max_scores)
                print("\n=================================================\n")

        if metric == "" or metric == "l1":
            # Evaluating max diversity using l1 distance:
            sim_dict = d_sim(dl_embeddings, q, metric="l1", normalize=normalize) # index the similarity between each item in S and q for once so that we can re-use them.
            div_dict = d_div(dl_embeddings, metric = "l1", normalize=normalize) # index the diversity between each pair of items in S so that we can re-use them.
            l1_with_query_max_scores = f_prime(R_with_query, lmda, k, sim_dict, div_dict)
            l1_wo_query_max_scores = f_prime(R_without_query, lmda, k, sim_dict, div_dict)
            
            if print_results == True:
                print("Evaluating max diversity using l1 distance:")
                print("max score with query: ", l1_with_query_max_scores)
                print("max score without query: ", l1_wo_query_max_scores)
                print("\n=================================================\n")
        if metric == "" or metric == "l2":
            # Evaluating max diversity using l2 distance:
            sim_dict = d_sim(dl_embeddings, q, metric="l2", normalize=normalize) # index the similarity between each item in S and q for once so that we can re-use them.
            div_dict = d_div(dl_embeddings, metric = "l2", normalize=normalize) # index the diversity between each pair of items in S so that we can re-use them.
            l2_with_query_max_scores = f_prime(R_with_query, lmda, k, sim_dict, div_dict)
            l2_wo_query_max_scores = f_prime(R_without_query, lmda, k, sim_dict, div_dict)            
        if print_results == True:
            print("Evaluating max diversity using l2 distance:")
            print("max score with query: ", l2_with_query_max_scores)
            print("max score without query: ", l2_wo_query_max_scores)
            print("\n=================================================\n")
    else:
        cosine_with_query_max_scores = np.nan
        cosine_wo_query_max_scores = np.nan
        l1_with_query_max_scores = np.nan
        l1_wo_query_max_scores = np.nan
        l2_with_query_max_scores = np.nan
        l2_wo_query_max_scores = np.nan
    if metric == "" or metric == "cosine":    
        # Evaluating max-min diversity and average distance using cosine distance:
        cosine_with_query_min_scores = min_div_score(final_ranking_with_query, metric="cosine", normalize=normalize)
        cosine_wo_query_min_scores = min_div_score(ranking_without_query, metric = "cosine", normalize=normalize)
        cosine_with_query_avg_scores =  sum(cosine_with_query_min_scores) / len(cosine_with_query_min_scores)
        cosine_wo_query_avg_scores = sum(cosine_wo_query_min_scores) / len(cosine_wo_query_min_scores)
        
        # The function below computes distance between pairs of data lake and data lake points, data lake and query points but not query and query points. 
        cosine_w_mix_query_min_scores = min_mix_div_score(final_ranking_with_query, set(query_embeddings.keys()), metric="cosine", normalize=normalize)
        cosine_w_mix_query_avg_scores =  sum(cosine_w_mix_query_min_scores) / len(cosine_w_mix_query_min_scores)

        if print_results == True:
            print ("Evaluating max-min diversity and average distance using cosine distance:")
            print("max-min score with query: ", min(cosine_with_query_min_scores))
            print("max-min score without query: ", min(cosine_wo_query_min_scores))
            print("max-min score with mix query: ", min(cosine_w_mix_query_min_scores))
            print("average distance with query: ", cosine_with_query_avg_scores)
            print("average distance without query: ", cosine_wo_query_avg_scores)
            print("average distance with mix query: ", cosine_w_mix_query_avg_scores)
            print("\n=================================================\n")
    if metric == "" or metric == "l1":
        # Evaluating max-min diversity and average distance using l1 distance:
        l1_with_query_min_scores = min_div_score(final_ranking_with_query, metric="l1", normalize=normalize)
        l1_wo_query_min_scores = min_div_score(ranking_without_query, metric = "l1", normalize=normalize)
        l1_with_query_avg_scores = sum(l1_with_query_min_scores) / len(l1_with_query_min_scores)
        l1_wo_query_avg_scores = sum(l1_wo_query_min_scores) / len(l1_wo_query_min_scores)
        
        # The function below computes distance between pairs of data lake and data lake points, data lake and query points but not query and query points. 
        l1_w_mix_query_min_scores = min_mix_div_score(final_ranking_with_query, set(query_embeddings.keys()), metric="l1", normalize=normalize)
        l1_w_mix_query_avg_scores =  sum(l1_w_mix_query_min_scores) / len(l1_w_mix_query_min_scores)

        if print_results == True:
            print ("Evaluating max-min diversity and average distance using l1 distance:")
            print("max-min score with query: ", min(l1_with_query_min_scores))
            print("max-min score without query: ", min(l1_wo_query_min_scores))
            print("max-min score with mix query: ", min(l1_w_mix_query_min_scores))
            print("average distance with query: ", l1_with_query_avg_scores)
            print("average distance without query: ", l1_wo_query_avg_scores)
            print("average distance with mix query: ", l1_w_mix_query_avg_scores)
            print("\n=================================================\n")
    
    if metric == "" or metric == "l2":
        # Evaluating max-min diversity and average distance using l2 distance:
        l2_with_query_min_scores = min_div_score(final_ranking_with_query, metric= "l2", normalize=normalize)
        l2_wo_query_min_scores = min_div_score(ranking_without_query, metric = "l2", normalize=normalize)
        l2_with_query_avg_scores = sum(l2_with_query_min_scores) / len(l2_with_query_min_scores)
        l2_wo_query_avg_scores = sum(l2_wo_query_min_scores) / len(l2_wo_query_min_scores)
        
        # The function below computes distance between pairs of data lake and data lake points, data lake and query points but not query and query points. 
        l2_w_mix_query_min_scores = min_mix_div_score(final_ranking_with_query, set(query_embeddings.keys()), metric="l2", normalize=normalize)
        l2_w_mix_query_avg_scores =  sum(l2_w_mix_query_min_scores) / len(l2_w_mix_query_min_scores)

        if print_results == True:
            print("Evaluating max-min diversity and average distance using l2 distance:")
            print("score with query: ", min(l2_with_query_min_scores))
            print("score without query: ", min(l2_with_query_min_scores))
            print("max-min score with mix query: ", min(l2_w_mix_query_min_scores))
            print("average distance with query: ", l2_with_query_avg_scores)
            print("average distance without query: ", l2_wo_query_avg_scores)
            print("average distance with mix query: ", l2_w_mix_query_avg_scores)
            print("\n=================================================\n")
    
    # create 6 dictionaries to store all the calculations
    if metric == "" or metric == "cosine": 
        computed_metrics.append({"metric": "cosine", "with_query" : "yes", "max_score": cosine_with_query_max_scores, "max-min_score": min(cosine_with_query_min_scores), "avg_score": cosine_with_query_avg_scores})
        computed_metrics.append({"metric": "cosine", "with_query": "no", "max_score": cosine_wo_query_max_scores, "max-min_score": min(cosine_wo_query_min_scores), "avg_score": cosine_wo_query_avg_scores})
        computed_metrics.append({"metric": "cosine", "with_query": "mix", "max_score": np.nan, "max-min_score": min(cosine_w_mix_query_min_scores), "avg_score": cosine_w_mix_query_avg_scores})
        

    if metric == "" or metric == "l1":
        computed_metrics.append({"metric": "l1", "with_query" : "yes", "max_score": l1_with_query_max_scores, "max-min_score": min(l1_with_query_min_scores), "avg_score": l1_with_query_avg_scores})
        computed_metrics.append({"metric": "l1", "with_query": "no", "max_score": l1_wo_query_max_scores, "max-min_score": min(l1_wo_query_min_scores), "avg_score": l1_wo_query_avg_scores})
        computed_metrics.append({"metric": "l1", "with_query": "mix", "max_score": np.nan, "max-min_score": min(l1_w_mix_query_min_scores), "avg_score": l1_w_mix_query_avg_scores})

    if metric == "" or metric == "l2":
        computed_metrics.append({"metric": "l2", "with_query" : "yes", "max_score": l2_with_query_max_scores, "max-min_score": min(l2_with_query_min_scores), "avg_score": l2_with_query_avg_scores})
        computed_metrics.append({"metric": "l2", "with_query": "no", "max_score": l2_wo_query_max_scores, "max-min_score": min(l2_wo_query_min_scores), "avg_score": l2_wo_query_avg_scores})
        computed_metrics.append({"metric": "l2", "with_query": "mix", "max_score": np.nan, "max-min_score": min(l2_w_mix_query_min_scores), "avg_score": l2_w_mix_query_avg_scores})

    return computed_metrics, embedding_plot
    
    

def mmc(s_set: set, lmda : float, k: int, sim_dict: dict, div_dict : dict, R_p: set) -> dict: # s_dict contains query id as key and its embeddings as values.
    all_mmc = dict()
    # print("R_P:", R_p)
    p = len(R_p) - 1
    div_coefficient = lmda / (k - 1)
    for s_i in s_set:
        sim_term = (1 - lmda) * sim_dict[s_i]
        div_term1 = div_coefficient * mmc_compute_div_sum(s_i, div_dict, R_p)
        div_term2 = div_coefficient * sum(mmc_compute_div_large(s_i, div_dict, s_set - R_p - {s_i}, k - p))
        current_mmc = sim_term + div_term1 + div_term2
        all_mmc[s_i] = current_mmc
    # print("current mmc:", current_mmc)
    # print("all_mmc:", all_mmc)
    return all_mmc

# Algorithms specific to GMC

def gmc(S_dict: dict, q_dict:dict, k: int, lmda: float = 0.7, metric = "cosine", print_results = False, normalize = False, max_metric = True, compute_metric = True) -> set: #S_dict is a dictionary with tuple id as key and its embeddings as value. 
    #the metric is for sim dict and div dict, and is independent of evaluation. we evaluate using all three metrics and in compute_metric() function, we again compute sim_dict and div_dict.
    start_time = time.time_ns()
    q = np.mean(list(q_dict.values()), axis=0)
    R = set()
    ranked_div_result = []
    sim_dict = d_sim(S_dict, q, metric = metric, normalize=normalize) # index the similarity between each item in S and q for once so that we can re-use them.
    div_dict = d_div(S_dict, metric = metric, normalize=normalize) # index the diversity between each pair of items in S so that we can re-use them.
    # debug_dict(sim_dict, 5, "sim dict")
    # debug_dict(div_dict, 5, "div dict")
    S_set = set(S_dict.keys())
    for p in range(0, k):
        if len(S_set) == 0:
            break
        mmc_dict = mmc(S_set, lmda, k, sim_dict, div_dict, R) # send S to mmc and compute MMC for each si
        s_i  = max(mmc_dict, key=lambda k: mmc_dict[k])
        R.add(s_i)
        ranked_div_result.append(s_i)
        S_set = S_set - {s_i}
    # print("GMC f score:", f_prime(R, lmda, k, sim_dict, div_dict))
    end_time = time.time_ns()
    total_time = round(int(end_time - start_time) / 10 ** 9, 2)
    print("Total time taken: ", total_time, " seconds.")
    if compute_metric == True:
        computed_metrics, embedding_plot = compute_metrics(R, S_dict, q_dict, lmda, k, print_results = print_results, normalize=normalize, metric= metric, max_metric= max_metric)
        for each in computed_metrics:
            each['time_taken'] = total_time
    else:
        computed_metrics = [{"metric": "n/a", "with_query" : "n/a", "max_score": np.nan, "max-min_score": np.nan, "avg_score": np.nan, 'time_taken' : total_time}]
        embedding_plot = ""
    return ranked_div_result, computed_metrics, embedding_plot


# Algorithms specific to GRASP
def gne_construction(s_set: set, lmda : float, k: int, alfa: float, sim_dict: dict, div_dict : dict,  randomize: bool, random_seed: int):
    R_p = set()
    for p in range(0, k):
        mmc_dict = mmc(s_set, lmda, k, sim_dict, div_dict, R_p)
        s_max = max(mmc_dict.values())
        s_min = min(mmc_dict.values())
        # print("S_MAX:", s_max)
        # debug_dict(mmc_dict, len(mmc_dict))
        # compute RCL
        RCL = set()
        threshold = s_max - alfa * (s_max - s_min)
        for s_i in mmc_dict:
            if mmc_dict[s_i] >= threshold:
                RCL.add(s_i)

        # using random seed value if we want the same result every time. It is still random but controlled randomization.
        if randomize == False:
            random_seed = random_seed
            random.seed(random_seed)

        s_i = random.choice(list(RCL))
        R_p.add(s_i)
        s_set = s_set - {s_i}
    # print(len(R_p))
    return R_p


def gne_local_search(gne_S_set: set, lmda: float, k: int, alfa: float, sim_dict: dict, div_dict: dict, gne_R: set):
    # print("start R: ", (gne_R))
    gne_R_dash = set()
    gne_R_dash = gne_R.copy()
    for s_i in gne_R:
        # print("Len R dash start:", len(gne_R_dash))
        s_i_l_list = gne_compute_div_large_key(s_i, div_dict, gne_S_set, k)
        for s_j in gne_R:
            if s_j == s_i:
                continue
            else:        
                for l in range(0, k-1):
                    s_i_l = s_i_l_list[l]
                    # print("SIL: ", s_i_l)
                    if s_i_l not in gne_R_dash:
                        R_dash_dash = set()
                        R_dash_dash = gne_R_dash.copy()
                        # print("R dash dash before len:", len(R_dash_dash))
                        if s_j in R_dash_dash:
                            R_dash_dash.remove(s_j)
                            R_dash_dash.add(s_i_l)
                        if f_prime(R_dash_dash, lmda, k, sim_dict, div_dict) > f_prime(gne_R_dash, lmda, k, sim_dict, div_dict):
                            gne_R_dash = set()
                            gne_R_dash = R_dash_dash.copy()
        # print("Len R dash end:", len(gne_R_dash))

    if f_prime(gne_R_dash, lmda, k, sim_dict, div_dict) > f_prime(gne_R, lmda, k, sim_dict, div_dict):
        gne_R = set()
        gne_R = gne_R_dash.copy()
    # print("End R: ",(gne_R))
    return gne_R




def grasp(S_dict: dict, q_dict:dict, k: int, i_max: int = 10, lmda: float = 0.7, alfa: float = 0.01, randomize: bool = True, random_seed: int = 42, metric = "cosine", print_results = False, normalize = False, max_metric = True, compute_metric = True) -> set: #S_dict is a dictionary with tuple id as key and its embeddings as value.
    #the metric is for sim dict and div dict, and is independent of evaluation. we evaluate using all three metrics and in compute_metric() function, we again compute sim_dict and div_dict.
    start_time = time.time_ns()
    q = np.mean(list(q_dict.values()), axis=0)
    R = set()
    r_f_prime = 0
    ranked_div_result = []
    sim_dict = d_sim(S_dict, q, metric = metric, normalize=normalize) # index the similarity between each item in S and q for once so that we can re-use them.
    div_dict = d_div(S_dict, metric = metric, normalize=normalize) # index the diversity between each pair of items in S so that we can re-use them.
    # debug_dict(sim_dict, 5, "sim dict")
    # debug_dict(div_dict, 5, "div dict")
    S_set = set(S_dict.keys())
    for i in range(0, i_max):
        R_dash = gne_construction(S_set, lmda, k, alfa, sim_dict, div_dict, randomize, random_seed)
        R_dash = gne_local_search(S_set, lmda, k, alfa, sim_dict, div_dict, R_dash) # note that we send R' which is received as R in the function, to reciprocate variable names used in the algorithm in the paper.
        r_dash_f_prime = f_prime(R_dash, lmda, k, sim_dict, div_dict)
        if r_dash_f_prime > r_f_prime:
            R = R_dash.copy()
            r_f_prime = r_dash_f_prime
    end_time = time.time_ns()
    total_time = round(int(end_time - start_time) / 10 ** 9, 2)
    print("Total time taken: ", total_time , " seconds.")
    if compute_metric == True:
        computed_metrics, embedding_plot = compute_metrics(R, S_dict, q_dict, lmda, k, print_results = print_results, normalize=normalize, metric=metric, max_metric = max_metric)
        for each in computed_metrics:
            each['time_taken'] = total_time
    else:
        computed_metrics = [{"metric": "n/a", "with_query" : "n/a", "max_score": np.nan, "max-min_score": np.nan, "avg_score": np.nan, 'time_taken' : total_time}]
        embedding_plot = ""
    return R, computed_metrics, embedding_plot



def cluster_tuples(embedding_dict, q_dict , k, method = "bkmeans", metric = "l2", lmda = 0.7, linkage = "average", helper_function = False, print_results = False, normalize =False, max_metric = True, compute_metric = True):
    start_time = time.time_ns()
    q = np.mean(list(q_dict.values()), axis=0)
    sim_dict = d_sim(embedding_dict, q, normalize=normalize) # index the similarity between each item in S and q for once so that we can re-use them.
    div_dict = d_div(embedding_dict, metric=metric, normalize=normalize) # index the diversity between each pair of items in S so that we can re-use them.
    # Extract embeddings as a numpy array
    embeddings = np.array(list(embedding_dict.values()))
    
    # Initialize with the desired number of clusters (k)
    if method == "kmeans":
        # Calculate the pairwise cosine distances (or similarities)
        distances = pairwise_distances(embeddings, metric=metric)
        kmeans = KMeans(n_clusters=k,  init='k-means++', random_state=0)
        kmeans.fit(distances)
        # Get the cluster assignments for each item
        cluster_assignments = kmeans.labels_
    elif method == "hierarchical":    
        # Initialize Agglomerative Clustering with the desired number of clusters (k) and 'cosine' metric
        agglomerative = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
        # Fit the Agglomerative Clustering model to the data
        cluster_assignments = agglomerative.fit_predict(embeddings)
    else:
        # Calculate the pairwise cosine distances (or similarities)
        distances = pairwise_distances(embeddings, metric=metric)
        bkmeans = BKMeans(n_clusters=k, random_state=0)
        bkmeans.fit(distances)
        # Get the cluster assignments for each item
        cluster_assignments = bkmeans.labels_
    
    # Get the centroids of each cluster
    cluster_centroids = {}
    for i in range(k):
        cluster_items_indices = np.where(cluster_assignments == i)[0]
        centroid = np.mean(embeddings[cluster_items_indices], axis=0)
        cluster_centroids[i] = centroid

    # Get the actual centroids (metoids) of each cluster
    cluster_metoids = {}
    for i in range(k):
        cluster_items_indices = np.where(cluster_assignments == i)[0]
        cluster_items = [list(embedding_dict.keys())[idx] for idx in cluster_items_indices]
        
        # Compute distances from each item to the centroid
        distances_to_centroid = np.sum((embeddings[cluster_items_indices] - cluster_centroids[i])**2, axis=1)
        # Find the item that is closest to the centroid
        closest_point_index = np.argmin(distances_to_centroid)
        cluster_metoids[i] = cluster_items[closest_point_index]
        #cluster_metoids[i] = cluster_items[np.argmin(np.sum((embeddings[cluster_items_indices] - embeddings[cluster_items_indices].mean(axis=0))**2, axis=1))]
    R = set(cluster_metoids.values())
    cluster_plot = visualize_clusters(embedding_dict, cluster_assignments, cluster_metoids, k, cluster_centroids, show_plot=print_results)
    end_time = time.time_ns()
    total_time = round(int(end_time - start_time) / 10 ** 9, 2)
    if helper_function == False:
        print("Total time taken: ", total_time , " seconds.")
        if compute_metric == True:
            computed_metrics, embedding_plot = compute_metrics(R, embedding_dict, q_dict, lmda, k, print_results = print_results, normalize= normalize, metric=metric, max_metric= max_metric)
            for each in computed_metrics:
                each['time_taken'] = total_time
        else:
            computed_metrics = [{"metric": "n/a", "with_query" : "n/a", "max_score": np.nan, "max-min_score": np.nan, "avg_score": np.nan, 'time_taken' : total_time}]
            embedding_plot = ""
        return R, computed_metrics, embedding_plot, cluster_plot
    else:
        return R, cluster_plot



def our_algorithm(embedding_dict, query_dict, k, method = "bkmeans", metric = "l2", lmda = 0.7, strategy = "average", linkage = "average", print_results = False, normalize = False, max_metric = True, compute_metric = True):
    start_time = time.time_ns()
    q = np.mean(list(query_dict.values()), axis=0)
    k_dash = min(len(embedding_dict), 2 * k)  # Number of clusters
    print("K dash = ", k_dash)
    cluster_metoids, cluster_plot = cluster_tuples(embedding_dict = embedding_dict, q_dict= query_dict, k = k_dash, method = method, linkage = linkage, metric = metric, helper_function= True, print_results= print_results, max_metric = max_metric, compute_metric = compute_metric)
    metoids_dict = {}
    for each in cluster_metoids:
        metoids_dict[each] = embedding_dict[each]
    # print("metoid dict:", sorted(metoids_dict.keys()))
    q_embeddings = list(query_dict.values())
    metoid_embeddings = list(metoids_dict.values())
    # print("metric:", metric)
    pairwise_distances_matrix = pairwise_distances(q_embeddings, metoid_embeddings, metric=metric)

    # Calculate the minimum, maximum, or average for each column
    if metric == "cosine":
        rev = True
    else:
        rev = True
    if strategy == "min":
        computed_values = np.min(pairwise_distances_matrix, axis=0)
        # rev = True
    elif strategy == "max":
        computed_values = np.max(pairwise_distances_matrix, axis=0)
        # rev = True
    else: #average
        computed_values = np.mean(pairwise_distances_matrix, axis=0)
        # rev = True

    key_value_pairs = list(zip(metoids_dict.keys(), computed_values))
    # print("k-v pairs", key_value_pairs)
    # Sort the list of pairs based on the selected strategy
    sorted_pairs = sorted(key_value_pairs, key=lambda x: x[1], reverse=rev)
    # Extract the sorted keys
    sorted_avg_keys = [pair[0] for pair in sorted_pairs][:k]
    end_time = time.time_ns()
    total_time = round(int(end_time - start_time) / 10 ** 9, 2)
    print("Total time taken: ", total_time , " seconds.")
    # print("Sorted avg keys:", sorted(sorted_avg_keys))
    if compute_metric == True:
        computed_metrics, embedding_plot = compute_metrics(sorted_avg_keys, embedding_dict, query_dict, lmda, k, print_results, normalize=normalize, metric=metric, max_metric = max_metric)
        for each in computed_metrics:
            each['time_taken'] = total_time
    else:
        computed_metrics = [{"metric": "n/a", "with_query" : "n/a", "max_score": np.nan, "max-min_score": np.nan, "avg_score": np.nan, 'time_taken' : total_time}]
        embedding_plot = ""
    return set(sorted_avg_keys), computed_metrics, embedding_plot, cluster_plot

