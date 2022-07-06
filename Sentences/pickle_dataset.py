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

rootdir = '/home/yasmin/Desktop/TG/Ep2/Curacion BIO'
layer='webanno.custom.NERClnico'

for file in os.listdir(rootdir):
    with open(rootdir+'/TypeSystem.xml','rb') as f:
        typesystem=load_typesystem(f)
        
    if file.endswith('xmi'):
        with open(rootdir+'/'+file,'rb')as f:
            cas=load_cas_from_xmi(f,typesystem=typesystem)
            data_hc, length= Label_anotations(file, cas, layer)
#print(length)
           
#find max, mean, and std length of sentences
sen_std = np.std(length)                         
sen_mean = np.mean(length)                       
sen_max = np.amax(length)                        
sen_precentile = np.percentile(length,99)  

with open('datahc_ner.pkl','wb') as f:
    pickle.dump(data_hc,f)