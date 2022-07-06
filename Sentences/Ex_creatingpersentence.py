#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:37:26 2022

@author: yasmin
"""

from cassis import *
import numpy as np
import pickle

rootdir = '/home/yasmin/Desktop/TG/Ep2/Curacion BIO'
layer='webanno.custom.NERClnico'
pickle_path="/home/yasmin/Desktop/TG/Ep3/datahc_ner.pkl"

data_hc={}
longitud=[]
Samples = {}

with open(rootdir+'/TypeSystem.xml','rb') as f:
    typesystem=load_typesystem(f)
    
for t in typesystem.get_types():
    print(f"{t.name}")
    
with open('/home/yasmin/Desktop/TG/Ep2/Curacion BIO/1134878.xmi','rb')as f:
    cas=load_cas_from_xmi(f,typesystem=typesystem)

listAnotations=[]


##Solamente selecciona las etiquetas
for data in cas.select(layer):
    listt=[[data.begin,data.end], data.Value, data.get_covered_text().split()] #format xmi
    listAnotations.append(listt)
    
TokensI = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
SentencesI="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
tokensf={}
spansf={}
tokens=[]
spans=[] #almacena los spans de inicio de cada palabra para hacer el etiquetado BIO
i=0

#Selecciona los tokens por cada sentencia
for t in cas.select(SentencesI):
    print(t)
    for s in cas.select_covered(TokensI,t):
        print(s)
        tokens.append(s.get_covered_text())
        spans.append(s.begin)
    i=i+1
    tokensf[i]=tokens
    spansf[i]=spans
    tokens=[]
    spans=[]

len(spansf[1]*2)


data_hcf={}
arreglo={}
samplef={}
labels={}
tokens_={}

for j in range(1,len(tokensf)+1):
    etiquetasbio=['O']*len(spansf[j])
    sample={}
    
    if len(listAnotations)==0:
        arreglo[j]=np.transpose(np.vstack((tokensf[i],etiquetasbio)))
        samplef[j]=arreglo[j]
        
    else:
        
        for anotacion in listAnotations:
            rangoS=anotacion[0]
            for item, i in enumerate(spansf[j]):
                if rangoS[0]== i:
                    etiquetasbio[item]=('B-' + str(anotacion[1]))
                
                elif rangoS[0] < i < rangoS[1]:
                    etiquetasbio[item]=('I-' + str(anotacion[1]))
                  
            labels[j]=list(etiquetasbio)
            tokens_[j]=list(tokensf[j])
        
        
            sample['sentences']= tokens_[j]
            sample['labels']= labels[j]
            data_hcf[j] = sample
            arreglo[j]=np.transpose(np.vstack((tokens_[j],labels[j])))

        sent_len=len(tokens_[j])
        longitud.append(sent_len)
        samplef[j]=arreglo[j]
        
data_hc[1]=data_hcf
Samples[1]=samplef #Diccionario con key: nombre del documento Value: tokens y su etiquetado

with open('datahc_ner.pkl','wb') as f:
    pickle.dump(data_hc,f)


with open(pickle_path,'rb') as f:
   data = pickle.load(f)

sentences = []
labels = []

for j in data:
    print(j)
    for v, (l,k) in enumerate(data[j].items()):
        print(k)
        sentences.append(k['sentences'])
        labels.append(k['labels'])
            