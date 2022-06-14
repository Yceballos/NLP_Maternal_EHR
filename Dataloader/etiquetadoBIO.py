# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:48:43 2022

@author: Dul
"""

from cassis import *
import numpy as np

Samples={}

def etiquetado(name, cas, layer):
    # name is the document name
    # cas is the document .xmi
    # layer is the layer of the annotations (we have two different annotations)

    ListAnotaciones=[]
    
    for data in cas.select(layer): #take entities from cas
        lista = [[data.begin, data.end], data.Value, data.get_covered_text().split()] #format xmi
        ListAnotaciones.append(lista)

    TokensI = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
    tokens=[]
    spans=[] #beginning spans to labbel BIO
    for s in cas.select(TokensI):
        tokens.append(s.get_covered_text())
        spans.append(s.begin) 
        
    etiquetasbio= ['O']*len(spans)
    
    if len(ListAnotaciones)==0:
        arreglo=np.transpose(np.vstack((tokens,etiquetasbio)))
        Samples[name]=arreglo
    
    else:
        for anotacion in ListAnotaciones:
           
           rangoS= anotacion[0] 
           for item, i in enumerate(spans):
               
              if rangoS[0]== i: #if el span_beginning= span(i) -> B-entity
                  etiquetasbio[item]=('B-' + str(anotacion[1]))
              
              elif rangoS[0] < i < rangoS[1]: #if el span(i) is contained on spans(inicio y final) -> I-entity
                  etiquetasbio[item]=('I-' + str(anotacion[1]))
              
           arreglo=np.transpose(np.vstack((tokens,etiquetasbio)))
        
        Samples[name]=arreglo #Dictionary
        
    return Samples, ListAnotaciones
