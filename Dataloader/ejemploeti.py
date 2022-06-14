# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 19:29:15 2022

@author: María Camila Durango 
"""
import os 
from cassis import *
from BIOlabel import label
import numpy as np
import pickle

for file in os.listdir():  # obtain listdir
    with open('TypeSystem.xml', 'rb') as f:
        typesystem = load_typesystem(f)
        
    if file.endswith(".xmi"): 
       with open(file, 'rb') as f:
           name=os.path.basename(file) #archive name
           cas= load_cas_from_xmi(f, typesystem=typesystem)
           
           #Hc=label(name,cas,'webanno.custom.Anonimizacin')
          # Hc,Di, oA=label(name,cas,'webanno.custom.NERClnico')
           data_hc, longitud=label(name,cas,'webanno.custom.NERClnico') 
           
#find max, mean, and std length of sentences
sen_std = np.std(longitud)                         
sen_mean = np.mean(longitud)                       
sen_max = np.amax(longitud)                        
sen_precentile = np.percentile(longitud,99)        
print(sen_std,sen_mean,sen_max,sen_precentile)

# save to disk
with open('datahc_ner.pkl','wb') as f:
    pickle.dump(data_hc,f)
   

           

    
           

          

           
