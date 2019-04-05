#!/usr/bin/env python
import os
import numpy as np
import h5py
from IPython import embed
# part 1
path_label = "/media/data_2/COCO_SIXD/linemod_test_kp/01/kp_label"
files= os.listdir(path_label)
listlabel = []
import random
random.shuffle(files)
for counter, file in enumerate(files):
    if counter % 1000==0: print(counter, "finished!")
    if counter > 186: break
    #generate 'part' of h5 file
    c = np.load(os.path.join(path_label,file))
    listlabel.append(c)
    files[counter] = file.replace("npy", "jpg")
parts = np.vstack(listlabel)
parts = np.reshape(parts,(-1,17,2))
print("parts: ", np.shape(parts))
# embed()

# part 2
path_rgb = "/media/data_2/COCO_SIXD/linemod_test_kp/01/rgb"
# files= os.listdir(path_rgb)
listnpy = []

for counter, file in enumerate(files):
    if counter % 1000==0: print(counter, "finished!")
    if counter > 186: break
    #generate 'imgname' of h5 file
    numblist = []
    for char in file:
        number = ord(char)
        numblist.append(number)
    tmparray = np.asarray(numblist)
    if len(numblist) == 16:    
        listnpy.append(tmparray)
    files[counter] = file.replace("jpg", "npy")    

imgnames = np.vstack(listnpy)
print("imgnames: ", np.shape(imgnames))
imgnum = np.shape(imgnames)[0]
# embed()

# part 3
path_bbox = "/media/data_2/COCO_SIXD/linemod_test_kp/01/bbox"
# files= os.listdir(path_bbox)
listbbox = []

for counter, file in enumerate(files):
    if counter % 1000==0: print(counter, "finished!")
    if counter > 186: break
    #generate 'bndbox' of h5 file
    c = np.load(os.path.join(path_bbox,file))
    listbbox.append(c)
bndboxes = np.vstack(listbbox)
bndboxes = np.reshape(bndboxes,(-1,1,4))
print("bndboxes: ", np.shape(bndboxes)) 
# embed()

with h5py.File("186_linemod_kp.h5", "w") as f:
    f.create_dataset("bndbox", data=bndboxes)
    f.create_dataset("imgname", data=imgnames)
    f.create_dataset("part", data=parts)

