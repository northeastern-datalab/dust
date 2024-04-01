import random
import glob
import os
import torch
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import _pickle as cPickle
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import utilities as utl
from transformers import BertTokenizer, BertModel
import torch, sys
import torch.nn as nn
from torch.nn.parallel import DataParallel
import prepare_dataset_utilities as prepare_utl
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union

def PreprocessDatasetSeparate(path):
    rows, labels = prepare_utl.LoadDatasetFromTSVFile(path)
    rows = [row.split("\t",1) for row in rows]
    return rows, labels
model = SentenceTransformer('bert-base-uncased') #case insensitive model. BOSTON and boston have the same embedding.
tokenizer = ""
start_time = time.time()
similarity_threshold = 0.7

benchmark_name = "tus_benchmark_corrected"
dataset_path = r"finetune_data" + os.sep + benchmark_name

train_path = dataset_path  + os.sep + "train"
valid_path = dataset_path  + os.sep + "valid"
test_path = dataset_path  + os.sep + "test"

train_dataset, train_labels = PreprocessDatasetSeparate(train_path)
valid_dataset, valid_labels = PreprocessDatasetSeparate(valid_path)
test_dataset, test_labels = PreprocessDatasetSeparate(test_path)

def EvaluateAccuracy(datasets, labels):
    correct_prediction = 0
    print("Total data points: ", len(datasets))
    for dataset, label in tqdm(zip(datasets, labels), desc = "Computing accuracy"):
        # print(dataset[0])
        # print(dataset[1])
        # print(label)
        embedding1 = model.encode(dataset[0])
        embedding2 = model.encode(dataset[1])
        # print(type(embedding1))
        # print(embedding2)
        cosine = utl.CosineSimilarity(embedding1, embedding2)
        predicted = 1 if cosine > similarity_threshold else 0
        if predicted == label:
            correct_prediction += 1
    return correct_prediction / len(labels)

print("Computing Train acc:")
train_acc = EvaluateAccuracy(train_dataset, train_labels)
print("Computing Valid acc:")
valid_acc = EvaluateAccuracy(valid_dataset, valid_labels)
print("Computing Test acc:")
test_acc = EvaluateAccuracy(test_dataset, test_labels)

print("Train acc: ", train_acc)
print("Valid acc: ", valid_acc)
print("Test acc: ", test_acc)