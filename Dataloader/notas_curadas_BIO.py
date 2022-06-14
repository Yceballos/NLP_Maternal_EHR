import os
from zipfile import ZipFile
import shutil

docs = os.walk("/home/yasmin/Desktop/TG/Curacion BIO/curation", topdown=False)
destination = "/home/yasmin/Desktop/TG/Curacion BIO"

arc_zip = []
name = []
for root, dirs, files in docs:
    for name in files:
       #print(name)
       if '.zip' in name:
           arc_zip.append(os.path.join(name))
    
    #%% PRUEBA   
    # print (os.getcwd()) # para saber en que direccion de trabajo estoy
    
    # contenido = os.chdir("C:/Users/Acer/Desktop/Curacion BIO/curation/{}".format(dirs[0]))
    
    # with ZipFile(arc_zip[0], 'r') as zip:
    #     zip.extractall()
        
    # with os.scandir(contenido) as ficheros:
    #     for fichero in ficheros:
    #         print(fichero.name)
    
    
    # now_name = os.path.join("C:/Users/Acer/Desktop/Curacion BIO/curation/{}".format(dirs[0]), 
    #                         "CURATION_USER.xmi")
    # new_name = os.path.join("C:/Users/Acer/Desktop/Curacion BIO/curation/{}".format(dirs[0]), 
    #                         "{}.xmi".format(dirs[0]))
    
    # shutil.move(now_name, new_name)
    
    
    # typesystem = 'TypeSystem.xml'
    # source = 'CURATION_USER_{}.xmi'.format(0)
    # destination = "C:/Users/Acer/Desktop/Curacion BIO"
    
    # shutil.move(source,destination)
    # shutil.move(typesystem,destination)

    #%% MELO
    for i in range(len(dirs)):
        contenido = os.chdir("/home/yasmin/Desktop/TG/Curacion BIO/curation/{}".format(dirs[i]))
        with ZipFile(arc_zip[i], 'r') as zip:
            zip.extractall()
            
        # with os.scandir(contenido) as ficheros:
        #     for fichero in ficheros:
        #         print(fichero.name)
        
        now_name = os.path.join("/home/yasmin/Desktop/TG/Curacion BIO/curation/{}".format(dirs[i]), 
                                "CURATION_USER.xmi")
        new_name = os.path.join("/home/yasmin/Desktop/TG/Curacion BIO/curation/{}".format(dirs[i]), 
                                "{}.xmi".format(dirs[i]))
        
        shutil.move(now_name, new_name)
        
        
        source = '{}.xmi'.format(dirs[i])
    
        shutil.move(source,destination)
    
    typesystem = 'TypeSystem.xml'
    shutil.move(typesystem,destination)