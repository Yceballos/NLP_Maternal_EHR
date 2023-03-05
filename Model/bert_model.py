# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 18:23:55 2022

@author: yasmin

"""

##test prueba 2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import logging
import random
import pickle
import argparse
import transformers
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertConfig, BertForTokenClassification,  AutoModelForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from nervaluate import Evaluator
from Processing import PretrainDataset
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# args
bert_path = "/home/yasmin/Desktop/NLP_Maternal_EHR/bsc-bio-ehr-es"# Directory containing BERT checkpoint
#vocab_file = "C:/Users/User/Desktop/es-cli-bert/bsc-bio-es" #json vocab file
pickle_path="/home/yasmin/Desktop/TG/Ep4-NewHCV/datahc_ner.pkl"
batch_size = 4 ###7max
patience = 2

##setup
device = torch.device("cpu") #is still in CPU!
n_gpu = torch.cuda.device_count()


# load data
dataset = PretrainDataset(pickle_path,
                          tokenizer_path=bert_path)  #objeto
                          
num_records = len(dataset)
num_classes = dataset.num_classes 


##Test train dataset
# train_size = int(num_records * 0.8)
# indices = np.arange(num_records)
# train_indices = indices[-train_size:]
# val_indices = indices[:-train_size]


np.random.seed(135568109) 
                     
train_indices = np.random.choice(range(num_records),int(0.8*num_records),replace=False)
val_indices  = np.asarray(list(set(range(num_records)) - set(train_indices)))

print(train_indices.shape) 
print(val_indices.shape)  

train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)



##Initialize BERT model
model= AutoModelForTokenClassification.from_pretrained(bert_path,num_labels=num_classes,output_hidden_states = True)

print(model)

model.train() ##11 capas de atención 
model.to(device)
params = [{'params':[p for n, p in model.named_parameters()],'weight_decay':0.001}]
optimizer = torch.optim.Adam(params,lr=5e-5)
best_val = 0
pat_counter = 0



# data loaders
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True) ##len 8 
val_loader = DataLoader(val_data,batch_size=1,shuffle=True)

# train loop
all_loss = {'train_loss':[], 'val_loss':[]}

# train loop
for ep in range(10):
    
    for i,batch in enumerate(train_loader):
        
        input_ids = batch['tokens'].to(device)
        segment_ids = batch['seg_ids'].to(device)
        input_mask = batch['masks'].to(device)
        labels = batch['labels'].to(device)
        doc_len = batch['doc_lens']
        
        
        outputs = model(input_ids,input_mask,segment_ids,labels=labels)
        #print("out: ",outputs)
        loss,scores =outputs[:2]
        #         loss_train,scores_train= outputs_train[:2]
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss = loss.cpu().detach().numpy()
        
        sys.stdout.write('epoch %i batch %i loss: %.6f      \r' % (ep+1,i+1,loss))
        sys.stdout.flush()
        
    all_loss['train_loss'].append(loss)
    
    print()
    val_preds = []
    val_labels = []
    hidden_states=[]
    
    # validation loop
    with torch.no_grad():
        for i,batch in enumerate(val_loader):
        
            input_ids = batch['tokens'].to(device)
            segment_ids = batch['seg_ids'].to(device)
            input_mask = batch['masks'].to(device)
            labels = batch['labels'].to(device)
            doc_len = batch['doc_lens'][0]
            
           # loss,scores = model(input_ids,input_mask,segment_ids,labels=labels)
            outputs = model(input_ids,input_mask,segment_ids,labels=labels)
            loss,scores=outputs[:2]
            
            
            scores = scores[0,:doc_len,:].cpu().data.numpy()
            scores = np.argmax(scores,1)
            text,preds = dataset.convert_bert_outputs(
                         input_ids.cpu().data.numpy()[0,:doc_len],scores)
            text,labels = dataset.convert_bert_outputs(
                          input_ids.cpu().data.numpy()[0,:doc_len],
                          labels[0,:doc_len].cpu().data.numpy())
            val_preds.append(preds)
            val_labels.append(labels)
                        
            loss = loss.cpu().detach().numpy()

                       
            sys.stdout.write('predicting batch %i      \r' % (i+1))
            sys.stdout.flush()
            
            hidden_states.append(outputs[2])
            
            
            
    all_loss['val_loss'].append(loss)

    #Validation score      
    evaluator = Evaluator(val_labels,val_preds,tags= ['Temporal Concept',
            'Sign or Symptom',
            'Quantitative Concept',
            'Qualitative Concept',
            'Body Part',
            'Disease or Syndrome',
            'Concept',
            'Idea or Concept',
            'Health Care Activity',
            'O'], loader="list")
    
    results, results_by_tag = evaluator.evaluate()
    
    exact_p = results['exact']['precision']
    exact_r = results['exact']['recall']
    exact_f = results['exact']['f1']
    partial_p = results['partial']['precision']
    partial_r = results['partial']['recall']
    partial_f = results['partial']['f1']
    
    print('epoch %i exact p: %.4f' % (ep+1,exact_p))
    print('epoch %i exact r: %.4f' % (ep+1,exact_r))
    print('epoch %i exact f: %.4f' % (ep+1,exact_f))
    
    if exact_f >= best_val:
        best_val = exact_f
        model.save_pretrained('savedmodels/bert_%s')
        pat_counter = 0
    else:
        pat_counter += 1
        if pat_counter >= patience:
            break

############################Embeddings####################33
token_embeddings = torch.stack(hidden_states, dim=0)

token_embeddings.size()
# Stores the token vectors, with shape [22 x 3,072]
token_vecs_cat = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# # For each token in the sentence...
# for token in token_embeddings:
    
#     # `token` is a [12 x 768] tensor

#     # Concatenate the vectors (that is, append them together) from the last 
#     # four layers.
#     # Each layer vector is 768 values, so `cat_vec` is length 3,072.
#     cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    
#     # Use `cat_vec` to represent `token`.
#     token_vecs_cat.append(cat_vec)
    
    

   # save best model vased on val score
plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

      
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(np.hstack(val_labels), np.hstack(val_preds), figsize=(12,15),x_tick_rotation=45, normalize=all)  
plt.show()


df=pd.DataFrame(np.hstack(val_labels), columns=["Labels"])
(df["Labels"].value_counts()/len(df))
fig=sns.countplot(x="Labels",data=df, orient="h", palette="inferno")
fig.set_xticklabels(fig.get_xticklabels(), rotation=45)

F1_score=f1_score(np.hstack(val_labels), np.hstack(val_preds), average='macro')
print('F1 score macro of: ',F1_score )



accuracy_score(np.hstack(val_labels), np.hstack(val_preds))
