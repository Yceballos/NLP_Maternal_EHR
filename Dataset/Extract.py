#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:00:52 2022

Extract information from files, only unzip files.
"""

import os
import shutil
rootdir = '/home/yasmin/Desktop/TG/Ep4-NewHCV/ehr-juan-parada/annotation'
extractdir='/home/yasmin/Desktop/TG/Ep4-NewHCV/ehr-juan-parada'

for file in os.listdir(rootdir):
    carpet=rootdir+'/'+file
    print(carpet)
    if not file.endswith('xml'):
        for rar in os.listdir(carpet):
            zipf=carpet+'/'+rar
            if rar.endswith('zip'):
                shutil.unpack_archive(zipf,extract_dir=(carpet))
                for file2 in os.listdir(carpet):
                    if 'juan.parada' in file2:
                        actualname=carpet+'/juan.parada.xmi'
                        #rename=extractdir+'/'+file+'.xmi'
                        rename=extractdir+'/'+file.split(".")[0]+'.xmi'
                        print(rename)
                        os.rename(actualname, rename)