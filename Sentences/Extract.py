#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:00:52 2022

Extract information from files, only unzip files.
"""

import os
import shutil
rootdir = '/home/yasmin/Desktop/TG/Ep2/Curacion BIO/curation'
extractdir='/home/yasmin/Desktop/TG/Ep2/Curacion BIO'

for file in os.listdir(rootdir):
    carpet=rootdir+'/'+file
    for rar in os.listdir(carpet):
        zipf=carpet+'/'+rar
        if rar.endswith('zip'):
            shutil.unpack_archive(zipf,extract_dir=(extractdir))
            actualname=extractdir+'/CURATION_USER.xmi'
            rename=extractdir+'/'+file+'.xmi'
            os.rename(actualname, rename)