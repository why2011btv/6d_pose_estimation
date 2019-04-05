#!/usr/bin/env python
import os
import numpy as np
import h5py
from IPython import embed

# part 1
path_label = "/media/data_2/COCO_SIXD/small_KP_linemod_10_24/01/kp_label"
files= os.listdir(path_label)
listlabel = []

for file in files:
    #generate 'part' of h5 file
    c = np.load(os.path.join(path_label,file))
    listlabel.append(c)
parts = np.vstack(listlabel)
parts = np.reshape(parts,(-1,17,2))
print("parts: ", np.shape(parts))

# part 2
path_rgb = "/media/data_2/COCO_SIXD/small_KP_linemod_10_24/01/rgb"
files= os.listdir(path_rgb)
listnpy = []

for file in files:
    #generate 'imgname' of h5 file
    numblist = []
    for char in file:
        number = ord(char)
        numblist.append(number)
    tmparray = np.asarray(numblist)

    if len(numblist) == 16:    
        listnpy.append(tmparray)
        embed()
        
imgnames = np.vstack(listnpy)
print("Shape of imgnames: ", np.shape(imgnames))
imgnum = np.shape(imgnames)[0]

# part 3
path_bbox = "/media/data_2/COCO_SIXD/small_KP_linemod_10_24/01/bbox"
files= os.listdir(path_bbox)
listbbox = []

for file in files:
    #generate 'bndbox' of h5 file
    c = np.load(os.path.join(path_bbox,file))
    listbbox.append(c)
bndboxes = np.vstack(listbbox)
bndboxes = np.reshape(bndboxes,(-1,1,4))
print("Shape of bndboxes: ", np.shape(bndboxes)) 

with h5py.File("mytestfile.h5", "w") as f:
    f.create_dataset("bndbox", data=bndboxes)
    f.create_dataset("imgname", data=imgnames)
    f.create_dataset("part", data=parts)
