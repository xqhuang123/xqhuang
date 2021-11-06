# -*- coding: utf-8 -*-

from __future__ import print_function
from data_preprocessing import get_batches
from model import get_model, plot_curve, model_evaluate, auc_plot
from class_activation_map import  get_img_array, make_gradcam_heatmap, save_and_display_gradcam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight
from tensorflow.keras.models import load_model
import numpy as np
import argparse


# Training settings
parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--train_path', type=str, default='...\\train\\', help="traning set path")
parser.add_argument('--valid_path', type=str, default='...\\val\\', help="validation set path")
parser.add_argument('--test_path', type=str, default='...\\test\\', help="testing set path")
parser.add_argument('--n', type=int, default=2, help="number of class")
parser.add_argument('--saved_path', type=str, default='...\\result\\', help="saved path")
parser.add_argument('--mod', type=str, default="inceptionv3",help='model archetecture')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.01, help='drop out ratio')
parser.add_argument('--optm', type=str, default="sgd", help='optimizer')
parser.add_argument('--patience_n', type=int, default=10, help='number of patience')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--epoch_n', type=str, default=30, help='number of epoch')

args = vars(parser.parse_args())

train_path=args["train_path"]
valid_path=args["valid_path"]
test_path=args["test_path"]
saved_path=args["saved_path"]
mod=args["mod"]
n_class=args["n"]
lr=args["lr"]
dropout=args["dropout"]
optm=args["optm"]
patience_n=args["patience_n"]
batch_size=args["batch_size"]
epoch_n=args["epoch_n"]
saved_nm=mod+"_bs"+str(batch_size)+"_"+optm+"_lr"+str(lr)+"_dropout"+str(dropout)

# resize image based on architecture
if mod in ["vgg16","vgg19","mobilenetv2","resnet50"]:
    size_x,size_y=224,224
elif mod=="inceptionv3":
    size_x,size_y=299,299  

#get train, validation and test batches
train_batches, valid_batches, test_batches=get_batches(train_path,valid_path,test_path,size_x,size_y)

#get model
model=get_model(mod,n_class,lr,dropout,optm)

#get class weights for imbalanced dataset
class_weights = class_weight.compute_class_weight('balanced', np.unique(train_batches.classes),train_batches.classes)
class_weights = dict(enumerate(class_weights))

#set early stopping if the validation accuracy doesnot improve for a number of epoch
early_stopping=EarlyStopping(monitor="val_accuracy",patience=patience_n,min_delta=0.01,mode="max")
checkpoint = ModelCheckpoint(saved_path+saved_nm+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#fit the model
history=model.fit(train_batches,validation_data=valid_batches,class_weight=class_weights,epochs=epoch_n,callbacks=[early_stopping,checkpoint],verbose=2)

#plot learning curve
plot_curve(history,saved_path,saved_nm)

#evaluate the model using testing set
pred_prob=model_evaluate(saved_path,saved_nm, test_batches) # get probablities
auc=auc_plot(test_batches,pred_prob,saved_path,saved_nm) # get auc

##class activation map
img_path="..\\test.tif"
#load saved model
saved_model=load_model(saved_path+saved_nm+".h5")

img_size=(size_x,size_y)
last_conv_layer_name="conv2d_5"
cam_path=saved_path+last_conv_layer_name+"_test_cam.tif"

#get image array
img_array = get_img_array(img_path, size=img_size)/255

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(img_array, saved_model, last_conv_layer_name)
save_and_display_gradcam(img_path, heatmap,cam_path)












