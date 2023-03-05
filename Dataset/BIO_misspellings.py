#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:20:40 2022

Libray to make anotations, asign labels and cut sentences
"""

from cassis import *
import numpy as np
from Noisy_Channel import Sentence_Corrector

TokensI = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
SentencesI="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

data_hc={}
length=[]
Samples = {}
Annotations=[]
Words=[]
    
def Label_anotations(name,cas,text,layer):
   
    dict_annotations={'https://uts.nlm.nih.gov/uts/umls/semantic-network/T079':'Temporal Concept', #time duration
                          'https://uts.nlm.nih.gov/uts/umls/semantic-network/T184':'Sign or Symptom',
                          'https://uts.nlm.nih.gov/uts/umls/semantic-network/T081':'Quantitative Concept', # dimensions, quantity or capacity of something using some unit of measure
                          'https://uts.nlm.nih.gov/uts/umls/semantic-network/T058':'Health Care Activity',#An activity of or relating to the practice of medicine or involving the care of patients.
                          'https://uts.nlm.nih.gov/uts/umls/semantic-network/T080':'Qualitative Concept', #A concept which is an assessment of some quality, rather than a direct measurement.
                          'https://uts.nlm.nih.gov/uts/umls/semantic-network/T023':'Body Part',#Body Part, Organ, or Organ Component
                          'https://uts.nlm.nih.gov/uts/umls/semantic-network/T047':'Disease or Syndrome', #A condition which alters or interferes with a normal process, state, or activity of an organism.
                          'https://uts.nlm.nih.gov/uts/umls/concept/C1518422':'Concept',
                          'https://uts.nlm.nih.gov/uts/umls/semantic-network/T078':'Idea or Concept',
                          '':'None'} #An abstract concept, such as a social, religious or philosophical concept.
    ListAnnotations=[]
    
    for data in cas.select(layer):
        #print(data.concepts)
        if data.concepts!=None:
            listt=[[data.begin,data.end], dict_annotations[data.concepts], data.get_covered_text().split()] #format xmi
            ListAnnotations.append(listt)
            Annotations.append(dict_annotations[data.concepts])
        
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
    
    Sentence_Corrector
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
                
                l= tokens_[j]
                new_sentence=[]
                
                #Correct sentence
                for i in list(range(3,len(l),3)):
                    a=l[i-3:i]
                    a=' '.join(a)
                    #print('a: ',a)
                    corrector = Sentence_Corrector(text)
                    b=corrector.return_best_sentence(a)
                    #print('b: ',b)
                    c=b[0].split()
                    new_sentence.extend(c)
                    
                sample['sentences']= new_sentence
                    
                sample['labels']= labels[j]
                data_hcf[j] = sample
                arreglo[j]=np.transpose(np.vstack((tokens_[j],labels[j])))
                  
                  #hice transpuesta porque creo que así es mas fácil de analizar los datos
                   
            Words.append(new_sentence)
                    
            sent_len=len(new_sentence)
            length.append(sent_len)
            samplef[j]=arreglo[j]
    
    
    data_hc[name]=data_hcf
    Samples[name]=samplef #Diccionario con key: nombre del documento Value: tokens y su etiquetado
    
    return data_hc, length, Annotations, Words