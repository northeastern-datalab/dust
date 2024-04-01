import random
import pickle
import bz2
import os
import sys
import csv
import numpy as np
import pandas as pd
import _pickle as cPickle
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import utilities as utl

# Function to create samples.
def CreateSampleDataPoints(table_pairs, nrows, label, table_location, separator = ",", pairs = "linear"):
    selected_data_points = set()
    skiprows = lambda i: i>0 and random.random() > 0.1
    skiprows = 0
    for item in table_pairs:
        if os.path.exists(table_location + os.sep + "query" + os.sep + item[0]) == True:
            table1 = pd.read_csv(table_location + os.sep + "query" + os.sep + item[0], skiprows= skiprows, sep = separator,  nrows=nrows, encoding = "latin-1", on_bad_lines = "skip")
        else:
            table1 = pd.read_csv(table_location + os.sep + "datalake" + os.sep + item[0], skiprows=skiprows, sep = separator, nrows=nrows, encoding = "latin-1", on_bad_lines = "skip")
        if os.path.exists(table_location + os.sep + "datalake" + os.sep + item[1]) == True:
            table2 = pd.read_csv(table_location + os.sep + "datalake" + os.sep + item[1], skiprows=skiprows, sep = separator, nrows=nrows, encoding = "latin-1", on_bad_lines = "skip")
        else:
            table2 = pd.read_csv(table_location + os.sep + "query" + os.sep + item[1], skiprows=skiprows, sep = separator, nrows=nrows, encoding = "latin-1", on_bad_lines = "skip")
        rows1 = table1.to_dict(orient='records')
        rows2 = table2.to_dict(orient="records")
        if pairs == "quadratic":
            for i in range(0,len(rows1)):
                for j in range(0, len(rows2)):
                    data_point = utl.SerializeRow(rows1[i]) + "\t" + utl.SerializeRow(rows2[j]) + "\t" + label
                    selected_data_points.add(data_point)
        else:
            for i in range(0,min(len(rows1), len(rows2))):
                data_point = utl.SerializeRow(rows1[i]) + "\t" + utl.SerializeRow(rows2[i]) + "\t" + label
                selected_data_points.add(data_point)
    return selected_data_points

# Save after creating the data points.
def SaveSampleDataPoints(file_path, data_points):
    file_pointer = open(file_path, encoding = "latin-1", mode="w")
    for item in data_points:
        file_pointer.write(item+"\n")
    file_pointer.close()

# Load data points.
def LoadSampleDataPoints(file_path):
    file_pointer = open(file_path, encoding="latin-1", mode="r")
    data_points = set()
    for data_point in file_pointer:
        data_point = data_point.split("\n")
        for each in data_point:
            data_points.add(each)
    return data_points

def SplitTrainTest(data_points, ratio, random_seed = 50):
    # Set the random seed
    random.seed(random_seed)
    num_test_samples = int(len(data_points) * ratio)
    random.shuffle(data_points)
    train_data = data_points[:-num_test_samples]
    test_data = data_points[-num_test_samples:]
    return train_data, test_data

def SplitTrainTestValid(data_points, ratio):
    test_ratio = ratio
    train_data_points, test_data_points = SplitTrainTest(data_points,test_ratio)
    valid_ratio = len(test_data_points) / len(train_data_points)
    train_data_points, valid_data_points = SplitTrainTest(train_data_points,valid_ratio)
    train_data_points = set(train_data_points)
    test_data_points = set(test_data_points)
    valid_data_points = set(valid_data_points)
    if len(test_data_points.intersection(train_data_points)) > 0:
        print("Leakage between test and train")
    elif len(test_data_points.intersection(valid_data_points)) > 0:
        print("Leakage between test and valid")
    elif len(valid_data_points.intersection(train_data_points)) > 0:
        print("Leakage between valid and train")
    else:
        print("Successful! No leakage found during splitting.")
    return train_data_points, test_data_points, valid_data_points

def CountClassLabelSize(data_points):
    count_positive = 0
    count_negative = 0
    for data in data_points:
        if data[1] == "1":
            count_positive += 1
        else:
            count_negative += 1
    return count_positive, count_negative

# save train test and valid sets
def SaveDatasetAsTSVFile(data_point, save_path):
    file = open(save_path, mode="w", encoding="latin-1")
    total_lines = len(data_point)
    current_line = 0
    for item in data_point:
        current_line += 1
        if current_line < total_lines:
            end = "\n"
        else:
            end = ""
        file.write(str(item[0])+"\t"+str(item[1])+ end)
    file.close()

# save train test and valid sets
def LoadDatasetFromTSVFile(save_path):
    file = open(save_path, mode="r", encoding="latin-1")
    X_labels = list()
    Y_labels = list() 
    for line in file:
        line = line.rsplit("\t",1)
        if len(line) == 2:
            X_labels.append(line[0])
            Y_labels.append(int(line[1].replace('\n', '')))
    return X_labels, Y_labels

# save train test and valid sets
# NOT USED 
def LoadDatasetFromTSVFileSeparate(save_path):
    file = open(save_path, mode="r", encoding="latin-1")
    X_labels = list()
    Y_labels = list() 
    for line in file:
        line = line.rsplit("\t",2)
        if len(line) == 3:
            X_labels.append((line[0],line[1]))
            Y_labels.append(int(line[2].replace('\n', '')))
    return X_labels, Y_labels