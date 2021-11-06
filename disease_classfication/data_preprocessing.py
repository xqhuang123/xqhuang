# -*- coding: utf-8 -*-

import glob
import os
import cv2
import numpy as np
import shutil
import Augmentor
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# root directory and directory name of each class
root_dir = "...\\dataset"
classes_dir = ["Disease", "Normal","Trace"]


# make a new directory if it does not exist
def make_dirs(new_dirnm):
    if not os.path.exists(new_dirnm):
        new_dir=os.makedirs(new_dirnm)
        
    return new_dir
    


# crop image, height is same(baed on intial visualization of the original image), width is cut by x1 from left and x2 from right
def crop_image(image_dir,crop_path,x1,x2):
    file_path_ls=glob.glob(os.path.join(image_dir,"*.tif"))# get the original image file list
    for img_path in file_path_ls:
        img_nm=img_path.split("\\")[-1][0:-4] #last 4 charactors are ".tif"
        img = cv2.imread(img_path) # read the file
        h, w = img.shape[0:2] # get the height and width from the shape            
        cropImg = img[0:h, int(0+x1):int(w-x2),:] # cut x1,x2 from left and right
        crop_imgnm="cropped_img_" + img_nm + ".tif" #cropped image name
        cv2.imwrite(os.join.path(crop_path,crop_imgnm), cropImg) # put the cropped images in the destination    


        
# define a function to split dataset into train, validation and test at the ratio of 0.6:0.2:0.2
def split_dataset(root_dir,classes_dir,train_dirnm,val_dirnm,test_dirnm):    
    val_ratio = 0.15
    test_ratio = 0.15    
    # make folders for train, validation and test set in the root directory
    for cls in classes_dir:
        train_cls_dir=make_dirs(os.path.join(root_dir,train_dirnm,cls))
        val_cls_dir=make_dirs(os.path.join(root_dir,val_dirnm,cls))
        test_cls_dir=make_dirs(os.path.join(root_dir,test_dirnm,cls))        
        src = os.path.join(root_dir + cls) # original folder to copy images from
        allFileNames = os.listdir(src) # get all files
        np.random.shuffle(allFileNames) # shuffle the files
        # split into train,val and test
        train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)* (1 - val_ratio - test_ratio)), 
                                                               int(len(allFileNames)* (1 - test_ratio))])
        # get the files for train,val and test
        train_FileNames = [src+'\\'+ name for name in train_FileNames.tolist()]
        val_FileNames = [src+'\\' + name for name in val_FileNames.tolist()]
        test_FileNames = [src+'\\' + name for name in test_FileNames.tolist()]
        # Copy files to the corresponding folder 
        for name in train_FileNames:
            shutil.copy(name, train_cls_dir)
        for name in val_FileNames:
            shutil.copy(name, val_cls_dir)
        for name in test_FileNames:
            shutil.copy(name, test_cls_dir)            
            
 
            
# image augmentation,input includes training and augmentated images path, ratio of samples to be augmentated
# (augmentation is only perfomred in training set)
def augment_images(train_cls_dir,output_dir,ratio): 
    file_ls=glob.glob(os.path.join(train_cls_dir,"*.tif"))    
    n_sample=len(file_ls)
    # passing the path of the image original directory and augmentated image destination
    p=Augmentor.Pipeline(train_cls_dir,output_dir)    
    # defining augmentation parameters 
    p.flip_left_right(probability=0.4) 
    p.flip_top_bottom(probability=0.8)    
    ## rotate the images
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)    
    #skew
    p.skew(0.4, 0.5)    
    # zoom in or zoom out images
    p.zoom(probability = 0.2, min_factor = 1.2, max_factor = 1.2)    
    p.sample(int(n_sample*ratio))    
    # combine the original images with augmentation data
    for file in file_ls:
        shutil.copy(file, output_dir)        



## get training, vlaidation, and testing set of images and labels
def get_batches(train_path,valid_path,test_path,size_x,size_y):
    #get train batch
    train_batches=ImageDataGenerator(rescale=1.0/255.).flow_from_directory(train_path,
                                                                           class_mode='categorical',
                                                                           target_size=(size_x, size_y),
                                                                           shuffle=True)
    #get validation batch 
    valid_batches=ImageDataGenerator(rescale=1.0/255.).flow_from_directory(valid_path, 
                                                                           class_mode='categorical',
                                                                           target_size=(size_x, size_y), 
                                                                           shuffle=False)
    #get test batch 
    test_batches=ImageDataGenerator(rescale=1.0/255.).flow_from_directory(test_path,
                                                                          class_mode='categorical',                                                                          
                                                                          target_size=(size_x, size_y),
                                                                          shuffle=False)
    
    
    return train_batches,valid_batches,test_batches


    


    
    
