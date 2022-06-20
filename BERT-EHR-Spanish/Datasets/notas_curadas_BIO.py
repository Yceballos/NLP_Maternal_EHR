import os
from zipfile import ZipFile
import shutil

docs = os.walk("C:/Users/User/Downloads/Curacion BIO/curation", topdown=False)

##Busca nombre de los arvhios en curation
arc_zip = []
name = []
for root, dirs, files in docs:
   for name in files:
       #print(name)
       if '.zip' in name:
           arc_zip.append(os.path.join(name))

#%% MELO
for i in range(len(dirs)):
    contenido = os.chdir("C:/Users/User/Downloads/Curacion BIO/curation/{}".format(dirs[i]))
    with ZipFile(arc_zip[i], 'r') as zip:
        zip.extractall()
        
        
    # with os.scandir(contenido) as ficheros:
    #     for fichero in ficheros:
    #         print(fichero.name)
    
    now_name = os.path.join("C:/Users/User/Downloads/Curacion BIO/curation/{}".format(dirs[i]), 
                            "CURATION_USER.xmi")
    new_name = os.path.join("C:/Users/User/Downloads/Curacion BIO/curation/{}".format(dirs[i]), 
                            "{}.xmi".format(dirs[i]))
    
    shutil.move(now_name, new_name)
    
    
    
    source = '{}.xmi'.format(dirs[i])
    destination = "C:/Users/User/Downloads/Curacion BIO"

    shutil.move(source,destination)
    
typesystem = 'TypeSystem.xml'
shutil.move(typesystem,destination)






