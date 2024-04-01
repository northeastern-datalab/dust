# %%
import random
import glob
import os
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import _pickle as cPickle
from numpy.linalg import norm
import utilities as utl
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, Sigmoid
from torch.optim import *
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, RobertaTokenizerFast, RobertaModel
from torch.nn.parallel import DataParallel
import prepare_dataset_utilities as prepare_utl
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from model_classes import BertClassifier, BertClassifierPretrained
# %%
# -------------------------------------------------------------------------------------------
# Step 1: Fine-tune BERT model to use for embedding the tables. Here, we feed rows separately.
# --------------------------------------------------------------------------------------------


def ComputeCosineEmbeddingLoss(embeddings1, embeddings2, labels, margin = 0.0):
    labels = (labels * 2.0) - 1.0  # Convert 0/1 labels to -1/1 targets
    similarity = F.cosine_similarity(embeddings1, embeddings2, dim=1)
    # try:
    loss = F.cosine_embedding_loss(embeddings1, embeddings2, labels, margin=margin)
    # except Exception as e:
    #     print(e)
    #     print(f'embedding1: {embeddings1}')
    #     print(f'embedding2: {embeddings2}')
    #     print(f'labels: {labels}')

    return loss, similarity


def DeleteSpecialTokens(rows_list):
    cleaned_rows = []
    for row in rows_list:
        cleaned_row = row.replace('COL', 'COL').replace("VAL", '\t')
        cleaned_rows.append(cleaned_row)
    return cleaned_rows

def RemoveColumnName(sentence):
    # print("sentence: ", sentence)
    # Split the string by "COL" token
    tokens = sentence.split("COL")
    # Remove text before "VAL"
    for i in range(1, len(tokens)):
        val_index = tokens[i].find("VAL")
        tokens[i] = tokens[i][val_index:]

    # Concatenate the remaining tokens into a string
    result = " ".join(tokens[1:])
    # Remove any leading or trailing whitespaces
    result = result.strip()
    # print("result: ", result)
    return result

# input_sentence = "COL column1 name VAL column1 value COL column2 name VAL column2 value COL column3 name VAL column3 value"
def UseSEPToken(sentence):
    # Split the input sentence into pairs of column name and value
    pairs = sentence.split('COL')[1:]
    # Create the transformed sentence
    transformed_sentence = "[CLS] " + " [SEP] ".join(" ".join(pair.strip().replace("VAL", "").split(" ")) for pair in pairs) + " [SEP]"
    transformed_sentence = transformed_sentence.strip()
    return transformed_sentence

#data preprocessing function
def PreprocessDatasetSeparate(path, data_usage, device, tokenizer_type = 'bert-base-uncased'):
    rows, labels = prepare_utl.LoadDatasetFromTSVFile(path)
    # rows = DeleteSpecialTokens(rows)
    rows = rows[0:int(len(rows) * data_usage)]
    labels = labels[0:int(len(labels) * data_usage)]
    rows = [row.split("\t",1) for row in rows]
    pos_labels = 0
    neg_labels = 0
    undefined_labels = 0
    for item in labels:
        if int(item) == 0:
            neg_labels += 1
        elif int(item) == 1:
            pos_labels += 1
        else:
            undefined_labels += 1
    print(f"Dataset: {path.rsplit(os.sep,1)[-1]}")
    print("Positive labels:", pos_labels)
    print("Negative labels:", neg_labels)
    print("Undefined labels:", undefined_labels)
    print("Total labels:", pos_labels + neg_labels + undefined_labels)
    # special_tokens_dict = {'additional_special_tokens': ['COL','VAL', '[COL]', '[VAL]']}
    custom_separator1 = "COL" 
    custom_separator2 = "VAL" 
    if tokenizer_type == "roberta-base":
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    else:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_type)
    # tokenizer.add_special_tokens({"additional_special_tokens": [custom_separator1, custom_separator2]})
    # tokenizer.add_special_tokens(special_tokens_dict)

    # Tokenize and encode the input sentences for the training set
    encodings_1 = tokenizer([UseSEPToken(pair[0]) for pair in rows], add_special_tokens = True, truncation = True, padding=True)
    encodings_2 = tokenizer([UseSEPToken(pair[1]) for pair in rows], add_special_tokens = True, truncation = True, padding=True)

    dataset = torch.utils.data.TensorDataset(torch.tensor(encodings_1['input_ids']).to(device), 
                                            torch.tensor(encodings_2['input_ids']).to(device),
                                            torch.tensor(encodings_1['attention_mask']).to(device),
                                            torch.tensor(encodings_2['attention_mask']).to(device),
                                            torch.tensor(labels).to(device))
    return dataset

# %%
#set parameters 
model_type = "roberta" # options: {"bert", "roberta"}
benchmark_name = "tus_finetune"
num_epochs = 100
pretrained = 0 # 1 means use pretrained model, not finetuned.
dataset_path = r"data/finetune_data" + os.sep + benchmark_name
out_dir = r"out_model" + os.sep + benchmark_name + "_" + model_type
if pretrained != 1:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(out_dir + os.sep + "checkpoints")
        os.makedirs(out_dir + os.sep + "plots")
        os.makedirs(out_dir + os.sep + "stats")
        print("Using output dir:", out_dir)
    else:
        print(f"Out directory {out_dir} already exists.")
        sys.exit()
batch_size = 16
learning_rate = 0.001
train_data_usage = 1
valid_data_usage = 1
test_data_usage = 1
similarity_threshold = 0.7
patience = 10
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
# %%
# Load pre-trained model and tokenizer
if model_type == "roberta":
    bert_model = RobertaModel.from_pretrained('roberta-base')
    print("Using RoBERTa model.")
    #roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
else:
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    print("Using BERT model.")

num_classes = 2
hidden_size = 768
output_size = 768
cosine_margin = 0.0
if pretrained == 1:
    model = BertClassifierPretrained(bert_model)
else:
    model = BertClassifier(bert_model, num_classes, hidden_size, output_size)

model = DataParallel(model, device_ids=[0, 1, 2, 3])
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5, verbose=True)
model.to(device)

# %%
# Prepare data for finetuning.
train_path = dataset_path  + os.sep + "train"
valid_path = dataset_path  + os.sep + "valid"
test_path = dataset_path  + os.sep + "test"

if model_type == "roberta":
    tokenizer_type = "roberta-base"
else: # default
    tokenizer_type = "bert-base-uncased"

train_dataset = PreprocessDatasetSeparate(train_path, train_data_usage, device, tokenizer_type=tokenizer_type)
valid_dataset = PreprocessDatasetSeparate(valid_path, valid_data_usage, device, tokenizer_type=tokenizer_type)
test_dataset = PreprocessDatasetSeparate(test_path, test_data_usage, device, tokenizer_type=tokenizer_type)


# %%
# Load data for finetuning
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training data size: {len(train_loader) * batch_size}")
print(f"Valid data size: {len(val_loader) * batch_size}")
print(f"Test data size: {len(test_loader) * batch_size}")

# %%
best_val_loss = float('inf')
best_val_epoch = 1
all_train_loss = []
all_test_loss = []
all_valid_loss = []
all_test_accuracy = []
all_valid_accuracy = []
all_train_accuracy = []
if pretrained != 1:
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')
        print('Epoch Start time:', time.strftime("%H:%M:%S"))
        running_loss = 0.0
        train_loss = 0.0
        start_time = time.time()
        model.train()
        train_acc = 0.0
        correct_predictions = 0
        num_predictions = 0
        for batch in tqdm(train_loader, desc = "Training"):
            input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, labels = batch
            input_ids_1 = input_ids_1.to(device)
            input_ids_2 = input_ids_2.to(device)
            attention_mask_1 = attention_mask_1.to(device)
            attention_mask_2 = attention_mask_2.to(device)
            labels = labels.to(device) 
            optimizer.zero_grad()
            outputs1 = model(input_ids_1, attention_mask_1)
            # print(f'outputs1: {outputs1}')
            outputs2 = model(input_ids_2, attention_mask_2)
            # print(f'outputs2: {outputs2}')
            loss, similarity_score = ComputeCosineEmbeddingLoss(outputs1, outputs2, labels)
            running_loss += loss.item()
            predicted = (similarity_score > similarity_threshold).long()
            num_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            # print(f'labels: {labels}')
            # print(f"predictions:{predicted}")
            # print(f"similarity: {similarity_score}")
            # print(f'loss: {loss}')
            loss.backward()
            optimizer.step()
            
        print('Epoch end time:', time.strftime("%H:%M:%S"))
        print(f'Epoch duration: {int(time.time()-start_time)/60:.4f} minutes.')
        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions/ num_predictions
        print(f'Train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
        scheduler.step(train_loss)

        # Evaluate on validation set
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        correct_predictions = 0
        num_predictions = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc = "Validating"):
                input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, labels = batch
                input_ids_1 = input_ids_1.to(device)
                input_ids_2 = input_ids_2.to(device)
                attention_mask_1 = attention_mask_1.to(device)
                attention_mask_2 = attention_mask_2.to(device)
                labels = labels.to(device)
                outputs1 = model(input_ids_1, attention_mask_1)
                outputs2 = model(input_ids_2, attention_mask_2)
                loss, similarity_score = ComputeCosineEmbeddingLoss(outputs1, outputs2, labels)
                val_loss += loss.item()
                predicted = (similarity_score > similarity_threshold).long()
                num_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()     
        val_loss /= len(val_loader)
        val_acc = correct_predictions/ num_predictions
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Evaluate on test set
        model.eval()
        test_loss , test_acc = 0.0, 0.0
        correct_predictions = 0
        num_predictions = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc = "Testing"):
                input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, labels = batch
                input_ids_1 = input_ids_1.to(device)
                input_ids_2 = input_ids_2.to(device)
                attention_mask_1 = attention_mask_1.to(device)
                attention_mask_2 = attention_mask_2.to(device)
                labels = labels.to(device)
                outputs1 = model(input_ids_1, attention_mask_1)
                outputs2 = model(input_ids_2, attention_mask_2)
                loss, similarity_score = ComputeCosineEmbeddingLoss(outputs1, outputs2, labels)
                test_loss += loss.item()
                predicted = (similarity_score > similarity_threshold).long()
                num_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()     
        test_loss /= len(test_loader)
        test_acc = correct_predictions/ num_predictions
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        # log statistics
        all_train_accuracy.append(train_acc)
        all_train_loss.append(train_loss)
        all_valid_loss.append(val_loss)
        all_test_loss.append(test_loss)
        all_valid_accuracy.append(val_acc)
        all_test_accuracy.append(test_acc)
        # continue
        
        # Save model checkpoint if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), out_dir + os.sep + "checkpoints" + os.sep + "best-checkpoint.pt")
            print(f'Saved checkpoint for epoch {epoch+1} with validation loss {val_loss:.4f}')

        
        # Save the most recent statistics
        utl.saveDictionaryAsPickleFile(all_train_accuracy, out_dir + os.sep + 'stats' + os.sep + 'all_train_accuracy.pickle')
        utl.saveDictionaryAsPickleFile(all_train_loss, out_dir + os.sep + 'stats' + os.sep + 'all_train_loss.pickle')
        utl.saveDictionaryAsPickleFile(all_test_loss, out_dir + os.sep + 'stats' + os.sep + 'all_test_loss.pickle')
        utl.saveDictionaryAsPickleFile(all_valid_loss, out_dir + os.sep + 'stats' + os.sep + 'all_valid_loss.pickle')
        utl.saveDictionaryAsPickleFile(all_valid_accuracy, out_dir + os.sep + 'stats' + os.sep + 'all_valid_accuracy.pickle')
        utl.saveDictionaryAsPickleFile(all_test_accuracy, out_dir + os.sep + 'stats' + os.sep + 'all_test_accuracy.pickle')

        # save the most recent plots
        loss_plot = {}
        loss_plot['Train'] = all_train_loss
        loss_plot['Test'] = all_test_loss
        loss_plot['Valid'] = all_valid_loss
        xlabel = "Epoch"
        ylabel = "Loss"
        title = "Separate serialization loss after different Epochs"
        figname = out_dir + os.sep + 'plots' + os.sep + 'loss_plot.jpg'
        utl.LinePlot(loss_plot, xlabel, ylabel, figname, title)

        accuracy_plot = {}
        accuracy_plot['Train'] = all_train_accuracy
        accuracy_plot["Test"] = all_test_accuracy
        accuracy_plot["Validation"] = all_valid_accuracy
        xlabel = "Epoch"
        ylabel = "Accuracy"
        title = "Test and validation accuracy after different Epochs"
        figname = out_dir + os.sep + 'plots' + os.sep + 'accuracy_plot.jpg'
        utl.LinePlot(accuracy_plot, xlabel, ylabel, figname,title)
        if epoch > (best_val_epoch - 1) + patience:
            print(f"Early stopped after {epoch + 1} epoch.")
            break

    # %%
    # Save the fine-tuned model
    torch.save(model.state_dict(), out_dir + os.sep + 'latest-checkpoint.pt')
    print(f'Saved final model and the best checkpoint from epoch {best_val_epoch}')

else: # compute on pretrained model
    # %%
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')
        print('Epoch Start time:', time.strftime("%H:%M:%S"))
        running_loss = 0.0
        train_loss = 0.0
        start_time = time.time()
        
        
        # Evaluate on test set
        model.eval()
        test_loss , test_acc = 0.0, 0.0
        correct_predictions = 0
        num_predictions = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc = "Test"):
                input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, labels = batch
                input_ids_1 = input_ids_1.to(device)
                input_ids_2 = input_ids_2.to(device)
                attention_mask_1 = attention_mask_1.to(device)
                attention_mask_2 = attention_mask_2.to(device)
                labels = labels.to(device)
                outputs1 = model(input_ids_1, attention_mask_1)
                outputs2 = model(input_ids_2, attention_mask_2)
                loss, similarity_score = ComputeCosineEmbeddingLoss(outputs1, outputs2, labels)
                test_loss += loss.item()
                predicted = (similarity_score > similarity_threshold).long()
                num_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()     
        test_loss /= len(test_loader)
        test_acc = correct_predictions/ num_predictions
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        # Evaluate on train set
        model.eval()
        train_acc = 0.0
        correct_predictions = 0
        num_predictions = 0
        with torch.no_grad():
            for batch in tqdm(train_loader, desc = "Training"):
                input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, labels = batch
                input_ids_1 = input_ids_1.to(device)
                input_ids_2 = input_ids_2.to(device)
                attention_mask_1 = attention_mask_1.to(device)
                attention_mask_2 = attention_mask_2.to(device)
                labels = labels.to(device) 
                outputs1 = model(input_ids_1, attention_mask_1)
                # print(f'outputs1: {outputs1}')
                outputs2 = model(input_ids_2, attention_mask_2)
                # print(f'outputs2: {outputs2}')
                loss, similarity_score = ComputeCosineEmbeddingLoss(outputs1, outputs2, labels)
                running_loss += loss.item()
                predicted = (similarity_score > similarity_threshold).long()
                num_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                # print(f'labels: {labels}')
                # print(f"predictions:{predicted}")
                # print(f"similarity: {similarity_score}")
                # print(f'loss: {loss}')
        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions/ num_predictions
        print(f'Train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
        
        # Evaluate on validation set
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        correct_predictions = 0
        num_predictions = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc = "Valid"):
                input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, labels = batch
                input_ids_1 = input_ids_1.to(device)
                input_ids_2 = input_ids_2.to(device)
                attention_mask_1 = attention_mask_1.to(device)
                attention_mask_2 = attention_mask_2.to(device)
                labels = labels.to(device)
                outputs1 = model(input_ids_1, attention_mask_1)
                outputs2 = model(input_ids_2, attention_mask_2)
                loss, similarity_score = ComputeCosineEmbeddingLoss(outputs1, outputs2, labels)
                val_loss += loss.item()
                predicted = (similarity_score > similarity_threshold).long()
                num_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()     
        val_loss /= len(val_loader)
        val_acc = correct_predictions/ num_predictions
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        
        break


