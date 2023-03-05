#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:56:50 2022

Extract data information from .xmi
"""

from cassis import *
import os
from BIOlabel import Label_anotations
import numpy as np
import pickle
import pandas as pd

rootdir = '/home/yasmin/Desktop/TG/Ep4-NewHCV/ehr-juan-parada'
layer='custom.Span'

for file in os.listdir(rootdir):
    with open(rootdir+'/TypeSystem.xml','rb') as f:
        typesystem=load_typesystem(f)
        
    if file.endswith('xmi'):
        with open(rootdir+'/'+file,'rb')as f:
            cas=load_cas_from_xmi(f,typesystem=typesystem)
            data_hc, length, Annotations, Words= Label_anotations(file, cas, layer)
#print(length)
           
#find max, mean, and std length of sentences
sen_std = np.std(length)                         
sen_mean = np.mean(length)                       
sen_max = np.amax(length)                        
sen_precentile = np.percentile(length,99)  

with open('datahc_ner.pkl','wb') as f:
    pickle.dump(data_hc,f)

pickle_path="/home/yasmin/Desktop/TG/Ep4-NewHCV/datahc_ner.pkl"
with open(pickle_path,'rb') as f:
    data = pickle.load(f)
    
pdAnnotations=pd.DataFrame(Annotations)
pdAnnotations=pdAnnotations.drop_duplicates().dropna()
print(pdAnnotations)

## Words statistics
import pandas as pd
df = pd.DataFrame(Words,columns=['words'])
dif_words=df.drop_duplicates()
print('Total words',df.size)
print('Unique words',dif_words.size)

##Counting words
word_frec = pd.DataFrame(columns=['word', 'freq'])
for word in dif_words['words']:
    #print(word)
    freq=Words.count(word)
    word_frec=word_frec.append({'word': word, 'freq':freq}, ignore_index=True)

word_frec = word_frec.sort_values('freq',ascending=False)
#Eliminamos caracteres especiales
word_frec=word_frec.drop([87, 86,141],axis=0)


##Counting Annotations
df = pd.DataFrame(Annotations,columns=['words'])
dif_annot=df.drop_duplicates()

annot_frec = pd.DataFrame(columns=['annot', 'freq'])
for word in dif_annot['words']:
    #print(word)
    freq=Annotations.count(word)
    annot_frec=annot_frec.append({'annot': word, 'freq':freq}, ignore_index=True)

annot_frec = annot_frec.sort_values('freq',ascending=False)

##Graph words - most comond n words
import matplotlib.pyplot as plt
from collections import Counter

def histogram(word_list, n):
  n = n # tengo que hacer esto para que el parámetro lo reconozca en .most_common(n).
  counts = dict(Counter(word_list).most_common(n))
  labels, values = zip(*counts.items())

  # ordenar los valores de manera descendiente
  indSort = np.argsort(values)[::-1]

  # reajustar los datos
  labels = np.array(labels)[indSort]
  values = np.array(values)[indSort]

  indexes = np.arange(len(labels))

  # configuración de la figura
  plt.figure(figsize = (15, 4))
  bar_width = 0.35
  plt.bar(indexes, values, color = 'b')

  # agrego las etiquetas
  plt.xticks((indexes + bar_width), labels, rotation = 'vertical')
  plt.tight_layout()
  plt.show()

histogram(Words, 20)

#Graph Annotations
import seaborn as sns
fig=sns.countplot(x="words",data=df, palette="inferno")
fig.set_xticklabels(fig.get_xticklabels(), rotation=45)