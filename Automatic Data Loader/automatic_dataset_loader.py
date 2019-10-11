# -*- coding: utf-8 -*-
"""Notes
1. creditcard.csv file has to be uploaded to the same folder to be able to use it!
2. Dataset names: cifar10 - cifar100 - creditcard - omniglot
3. For prototypical: cifar10_proto, omniglot_proto

"""
from keras.datasets import cifar10, cifar100
from sklearn.model_selection import train_test_split
import numpy as np
import keras 
import pandas as pd
import os
import imageio
import numpy as np
import shutil
import zipfile
import wget
from PIL import Image

path = os.path.join(os.path.dirname(os.path.realpath('__file__')))
os.chdir(path)
#os.chdir("C:/Users/falcoxman2/Desktop/Hildesheim/Master/Project/Repository/alperen/automated_dataset_loader")

def get_dataset(ds_name, normalize,ratio):
    
      print("Dataset:",ds_name)  
      if ds_name == 'cifar10':
        x_train, x_val, x_test, y_train, y_val, y_test = get_cifar10(ratio)
        normalize = True
      elif ds_name == 'cifar100':
        x_train, x_val, x_test, y_train, y_val, y_test = get_cifar100(ratio)
        normalize = True
      elif ds_name == 'creditcard':   
        x_train, x_val, x_test, y_train, y_val, y_test = get_creditcard(ratio)
      elif ds_name == 'omniglot':   
        x_train, x_val, x_test, y_train, y_val, y_test = get_omniglot(ratio)
        
        x_train=np.expand_dims(x_train, axis=3)
        x_val=np.expand_dims(x_val, axis=3)
        x_test=np.expand_dims(x_test, axis=3)
              
      elif ds_name == 'cifar10_proto':   
        x_train, x_val, x_test, y_train, y_val, y_test = get_cifar10_proto(ratio)
      elif ds_name == 'omniglot_proto':   
        x_train, x_val, x_test, y_train, y_val, y_test = get_omniglot_proto(ratio)
        y_train = np.arange(len(y_train)).tolist()
        y_val = np.arange(len(y_train),len(y_train)+len(y_val)).tolist()
        y_test = np.arange(len(y_train)+len(y_val),len(y_train)+len(y_val)+len(y_test)).tolist()
           
      if ds_name != 'omniglot_proto':      
      # Reshaping Targets/Classes
        print('omniglot_proto')
        num_classes = int(np.max([np.max(y_train),np.max(y_val),np.max(y_test)])+1)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
  
      #Normalizing Features for cifar datasets
      if normalize:
          x_train, x_val, x_test = normalize_data(ds_name, x_train, x_val, x_test)
    
      return x_train, x_val, x_test, y_train, y_val, y_test
    
def get_cifar10(ratio):
      (x_train, y_train), (x_test, y_test) = cifar10.load_data()
      X = np.concatenate((x_train, x_test), axis=0)
      y = np.concatenate((y_train, y_test), axis=0)
            
      return train_val_test_splitter(X, y, ratio, random_state=999)

def get_cifar100(ratio):
      (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
      X = np.concatenate((x_train, x_test), axis=0)
      y = np.concatenate((y_train, y_test), axis=0)
        
      return train_val_test_splitter(X, y, ratio, random_state=999)       

def get_creditcard(ratio):
      data = pd.read_csv('creditcard.csv')
      X = data.loc[:, data.columns != 'Class'].values
      y = data.iloc[:,-1].values
      y = y.reshape((len(y), 1))
      
      return train_val_test_splitter(X, y, ratio,  random_state=999)

def train_val_test_splitter(X, y, ratio, random_state=999):
      x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=999)
      x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=ratio/(1-ratio), random_state=999)
    
      return x_train, x_val, x_test, y_train, y_val, y_test


def normalize_data(ds_name, x_train, x_val, x_test):
      if ds_name == 'cifar10' or ds_name == 'cifar100':
            x_train = x_train/255
            x_val = x_val/255
            x_test = x_test/255
      return x_train, x_val, x_test
  
#--- Omniglot ---
def unzip():
  with zipfile.ZipFile('omniglot-master.zip', 'r') as zip_ref:
    zip_ref.extractall()
  with zipfile.ZipFile('omniglot-master/python/images_evaluation.zip', 'r') as zip_ref:
    zip_ref.extractall()
  with zipfile.ZipFile('omniglot-master/python/images_background.zip', 'r') as zip_ref:
    zip_ref.extractall()

#Reads and converts images to array    
def parse_images(data):
  images = []
  for img in data:
    im = imageio.imread(img)
    images.append(im)
  return images

#Reads and converts images to 28*28 array format 
def parse_images_resize(data):
  images = []
  for img in data:
    im = Image.open(img)
    im = im.resize((28,28), Image.ANTIALIAS)
    im = np.asarray(im, dtype="int32" )
    images.append(im)
  return images

#Cleans image files and zip after getting arrays
def clean():
  os.remove('omniglot-master.zip')
  shutil.rmtree('omniglot-master')
  shutil.rmtree('images_evaluation')
  shutil.rmtree('images_background')
 
def get_omniglot(ratio):
  
  # Download dataset from GitHub Repo
  url = 'https://github.com/brendenlake/omniglot/archive/master.zip'
  wget.download(url)
  # Unzip dataset
  unzip()  
  count = 0
  
  # Wrap all images
  alphabets, letters, labels = [], [], []   
  for file in os.listdir("images_background"):
      alphabets.append(os.path.join("images_background", file))
  for file in os.listdir("images_evaluation"):
      alphabets.append(os.path.join("images_evaluation", file))
    
  for alpha in alphabets:
    for file in os.listdir(alpha+'/'):
      
      path = os.path.join(alpha, file)
      
      for f in os.listdir(path):
        letters.append(path+'/'+f)
        labels.append(int(count))
        
      count += 1
  # Convert PNGs to arrays
  #images = parse_images(letters)  #Omniglot 105*105
  images = parse_images_resize(letters)   #Omniglot 28*28
  # Clean Base Dir from downloads
  clean()
  
  return  train_val_test_splitter(np.array(images), np.array(labels), ratio, random_state=999) 

#For Prototypical Networks
def get_cifar10_proto(ratio):
      (x_train, y_train), (x_test, y_test) = cifar10.load_data()
      X = np.concatenate((x_train, x_test), axis=0)
      y = np.concatenate((y_train, y_test), axis=0)
      
      X_all , y_all =[] , []
      for i in range(np.max(y)+1):
          data=X[np.where(y==i)[0]]
          X_all.append(data)
          y_all.append(np.repeat(i, 1000))
      
      return train_val_test_splitter(np.array([X_all])[0], np.array([y_all])[0], ratio, random_state=999)

def get_omniglot_proto(ratio):
  
  # Download dataset from GitHub Repo
  url = 'https://github.com/brendenlake/omniglot/archive/master.zip'
  wget.download(url)
  # Unzip dataset
  unzip()  
  count = 0
  
  # Wrap all images
  alphabets, letters, labels = [], [], [] 
  new_letters, new_labels = [], [] 
  for file in os.listdir("images_background"):
      alphabets.append(os.path.join("images_background", file))
  for file in os.listdir("images_evaluation"):
      alphabets.append(os.path.join("images_evaluation", file))
    
  for alpha in alphabets:
    for file in os.listdir(alpha+'/'):
      
      path = os.path.join(alpha, file)
      
      for f in os.listdir(path):
        letters.append(path+'/'+f)
        labels.append(int(count))
      
      new_letters.append(letters)
      new_labels.append(labels)
      count += 1
       
  # Convert PNGs to arrays
  #images = parse_images(letters)  #Omniglot 105*105
  images = parse_images_resize(new_letters[0])   #Omniglot 28*28
  images2=np.array(images).reshape((1623,20, 28,28))
  labels2=np.arange(1623)
  # Clean Base Dir from downloads
  clean()
  
  return  train_val_test_splitter(images2, labels2, ratio, random_state=999) 

#Run this to get train-val-test sets (For prototypical: cifar10_proto, omniglot_proto)
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('omniglot_proto',False,0.15)

x_train.shape, x_val.shape, x_test.shape
#y_train.shape, y_val.shape, y_test.shape

#For omniglot-prototypical
train_dataset = x_train
train_classes = y_train
val_dataset = x_val
val_classes = y_val
test_dataset = x_test
test_classes = y_test
n_classes = len(train_classes)
n_val_classes = len(val_classes)
n_test_classes = len(test_classes)

    
    