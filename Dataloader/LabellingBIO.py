# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:27:45 2022

@author: User
"""

##Sentenceias 

from cassis import *
import numpy as np

Samples={}
data_hc={}
arreglos=[]
longitud=[]

def label(name, cas, layer):
    # name is the document name
    # cas is the document .xmi
    # layer is the layer of the annotations (we have two different annotations)

    ListAnotaciones=[]
    
    for data in cas.select(layer): ##take entities from cas
        lista = [[data.begin, data.end], data.Value, data.get_covered_text().split()]
        ListAnotaciones.append(lista)
        
     
    TokensI = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
    tokens=[]

    spans=[] #take beginning spans for BIO
    for s in cas.select(TokensI):
        tokens.append(s.get_covered_text())
        spans.append(s.begin) 
        
    BIOlabel= ['O']*len(spans)
    
    if len(ListAnotaciones)==0:
        arra=np.transpose(np.vstack((tokens,BIOlabel)))
        Samples[name]=arra
    
    else:
        for anotation in ListAnotaciones:
           
           rangoS= anotation[0] 
           for item, i in enumerate(spans):
               
              if rangoS[0]== i: #if el span_beginning= span(i) -> B-entidad
                  BIOlabel[item]=('B-' + str(anotation[1]))
              
              elif rangoS[0] < i < rangoS[1]: #if el span(i) is contained on spans(inicio y final) -> I-entidad 
                  BIOlabel[item]=('I-' + str(anotation[1]))
             
        arra=np.transpose(np.vstack((tokens,BIOlabel)))
        sample = {}
        for i,(s,l) in enumerate(zip(tokens,BIOlabel)):
            sample['sentence']= tokens
            sample['labels']= BIOlabel
            data_hc[name] = sample
            sent_len=len(tokens)
        
        longitud.append(sent_len)
        
        Samples[name]=arra #Dictionary
       
   # return Samples, ListAnotaciones, tokens
    return data_hc, longitud  
