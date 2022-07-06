#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:20:40 2022

Libray to make anotations, asign labels and cut sentences
"""

from cassis import *
import numpy as np

TokensI = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
SentencesI="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

data_hc={}
length=[]
Samples = {}
    
def Label_anotations(name,cas,layer):
    # name is the document name
    # cas is the document .xmi
    # layer is the layer of the annotations (we have two different annotations)
    
    ListAnnotations=[]

    
    for data in cas.select(layer):
        listt=[[data.begin,data.end], data.Value, data.get_covered_text().split()] #format xmi
        ListAnnotations.append(listt)
    
    tokensf={}
    spansf={}
    tokens=[]
    spans=[] #almacena los spans de inicio de cada palabra para hacer el etiquetado BIO
    i=0

    #Selecciona los tokens por cada sentencia
    for t in cas.select(SentencesI):
        for s in cas.select_covered(TokensI,t):
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
        
        if len(ListAnnotations)==0:
            arreglo[j]=np.transpose(np.vstack((tokensf[i],etiquetasbio)))
            samplef[j]=arreglo[j]
            
        else:
            
            for anotacion in ListAnnotations:
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
              
               #hice transpuesta porque creo que así es mas fácil de analizar los datos
            sent_len=len(tokens_[j])
            length.append(sent_len)
            samplef[j]=arreglo[j]
            
    data_hc[name]=data_hcf
    Samples[name]=samplef #Diccionario con key: nombre del documento Value: tokens y su etiquetado
    
    return data_hc, length