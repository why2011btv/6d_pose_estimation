# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
import h5py
from functools import reduce

import torch.utils.data as data
from ..pose import generateSampleBox
from opt import opt
from IPython import embed

class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        # linemod_flag: 0 for step 1 ; 1 for step 2 or 3
        self.linemod_flag = 0
        self.seq_num = '15'
        # step 2 or 3
        if self.linemod_flag:
            if train:
                self.img_folder = '/u01/why/COCO_SIXD/linemod_test_50kp_11_3/linemod_test_50kp_11_3/' + self.seq_num + '/train'
            else: # validation dataset
                self.img_folder = '/u01/why/COCO_SIXD/linemod_test_50kp_11_3/linemod_test_50kp_11_3/' + self.seq_num + '/eval'
        
        # step 1
        else:
            self.img_folder = '/u01/why/COCO_SIXD/50_KP_11_1/' + self.seq_num + '/images'

        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = opt.nClasses
        self.nJoints = opt.nClasses

        self.accIdxs = tuple([i for i in range(1, opt.nClasses + 1)])
        self.flipRef = ()
        
        # step 2 or 3
        if self.linemod_flag:
            if train:
                filepath = '/u01/why/COCO_SIXD/linemod_test_50kp_11_3/linemod_test_50kp_11_3/' + self.seq_num + '/annot_train.h5'
                with h5py.File(filepath, 'r') as annot:
                    # embed()
                    # Modified by zelinzhao, original[:5887]
                    # train
                    self.imgname_coco_train = annot['imgname'][:]
                    self.bndbox_coco_train = annot['bndbox'][:]
                    self.part_coco_train = annot['part'][:]
            else:
                filepath = '/u01/why/COCO_SIXD/linemod_test_50kp_11_3/linemod_test_50kp_11_3/' + self.seq_num + '/annot_eval.h5'
                with h5py.File(filepath, 'r') as annot:
                    # val
                    self.imgname_coco_val = annot['imgname'][:]
                    self.bndbox_coco_val = annot['bndbox'][:]
                    self.part_coco_val = annot['part'][:]            
            if train:
                self.size_train = self.imgname_coco_train.shape[0]
            else:
                self.size_val = self.imgname_coco_val.shape[0]
                
        # step 1        
        else:
            filepath = '/u01/why/COCO_SIXD/50_KP_11_1/' + self.seq_num + '/annot_coco.h5'
            with h5py.File(filepath, 'r') as annot:
                # train
                self.imgname_coco_train = annot['imgname'][:-5887]
                self.bndbox_coco_train = annot['bndbox'][:-5887]
                self.part_coco_train = annot['part'][:-5887]
                # val
                self.imgname_coco_val = annot['imgname'][-5887:]
                self.bndbox_coco_val = annot['bndbox'][-5887:]
                self.part_coco_val = annot['part'][-5887:]

            self.size_train = self.imgname_coco_train.shape[0]
            self.size_val = self.imgname_coco_val.shape[0]
            
        '''
        if train:
            # filepath = os.path.join("/media/data_2/COCO_SIXD/50_KP_11_1/05/", "annot_coco.h5") #step1
            filepath = os.path.join("/media/data_2/COCO_SIXD/linemod_test_50kp_11_3/08/", "annot_train.h5")
            with h5py.File(filepath, 'r') as annot:
                # embed()
                # Modified by zelinzhao, original[:5887]
                # train
                self.imgname_coco_train = annot['imgname'][:]
                self.bndbox_coco_train = annot['bndbox'][:]
                self.part_coco_train = annot['part'][:]
        else:
            filepath = os.path.join("/media/data_2/COCO_SIXD/linemod_test_50kp_11_3/08/", "annot_eval.h5")
            with h5py.File(filepath, 'r') as annot:
                # val
                self.imgname_coco_val = annot['imgname'][:]
                self.bndbox_coco_val = annot['bndbox'][:]
                self.part_coco_val = annot['part'][:]            
        if train:
            self.size_train = self.imgname_coco_train.shape[0]
        else:
            self.size_val = self.imgname_coco_val.shape[0]
        '''

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            part = self.part_coco_train[index]
            bndbox = self.bndbox_coco_train[index]
            imgname = self.imgname_coco_train[index]
        else:
            part = self.part_coco_val[index]
            bndbox = self.bndbox_coco_val[index]
            imgname = self.imgname_coco_val[index]

        # lambda: short form of a function; map: in every ele of imgname
        imgname = reduce(lambda x, y: x + y,
                         map(lambda x: chr(int(x)), imgname))
        # print(imgname)
        img_path = os.path.join(self.img_folder, imgname)

        metaData = generateSampleBox(img_path, bndbox, part, self.nJoints,
                                     'coco', sf, self, train=self.is_train)

        inp, out, setMask = metaData
        # print("setMask shape is ", setMask.shape)
        return inp, out, setMask, 'coco'

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val
