import random
import glob
import os
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from numpy.linalg import norm
import utilities as utl
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, Sigmoid
from torch.optim import *
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, RobertaTokenizerFast, RobertaModel

# define the network

class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_labels, hidden_size, output_size, hidden_dropout_prob = 0.1):
        super().__init__()
        self.bert = bert_model
        # Freeze/ unfreeze all the parameters in the BERT model
        for param in self.bert.parameters():
            param.requires_grad = False
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.linear1 = nn.Linear(self.hidden_size, self.output_size)
        # self.lstm = nn.LSTM(self.output_size, self.output_size, batch_first=True)
        # self.linear2 = nn.Linear(self.output_size, self.output_size)
        # self.linear3 = nn.Linear(self.output_size, self.output_size)
        self.embed = nn.Linear(self.output_size, self.hidden_size)
        # self.classifier = nn.Linear(self.output_size, self.num_labels)
        
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = outputs.pooler_output # outputs[1]
        # return pooled_output # use this for pre-trained model result
        pooled_output = self.dropout(pooled_output)
        # pooled_output = nn.Tanh()(pooled_output)
        linear_output1 = nn.Tanh()(self.linear1(pooled_output))
        # linear_output1 = self.linear1(pooled_output)
        # lstm_output = self.lstm(linear_output1)[0]
        # linear_output2 = nn.Tanh()(self.linear2(linear_output1))
        # linear_output3 = nn.Tanh()(self.linear2(linear_output2))
        # linear_output1 = nn.Tanh()(linear_output1)
        embeddings = self.embed(linear_output1)
        # embeddings = linear_output1
        return embeddings

class BertClassifierPretrained(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        # Freeze/ unfreeze all the parameters in the BERT model
        for param in self.bert.parameters():
            param.requires_grad = False
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = outputs.pooler_output # outputs[1]
        return pooled_output # use this for pre-trained model result

