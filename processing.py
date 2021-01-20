# preprocessing
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb , gray2rgb
from skimage.transform import resize
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from pickle import dump
from pickle import load
from keras.utils import plot_model
from keras.models import load_model




def PrepareData(datapath , save_file , target_size , batch_size , feature_extract_model):
  path = "/content/data"
  index = 0
  train_datagen = ImageDataGenerator(rescale = 1. / 255)
  train = train_datagen.flow_from_directory(path, target_size = target_size , batch_size = batch_size , class_mode=None)
  for img in tqdm(train[0]):
    x = rgb2lab(img) #make input confirm to VGG16 input format
    data = dict()
    data["y_color_lab"] = x[: , : , 1: ] /128
    data['X_gray_lab'] =  x[: , : , 0:1] 
    x = gray2rgb(x[: , : , 0])
    x = x.reshape((1 ,) + x.shape)
    fc2_features = feature_extract_model.predict(x) 
    data['X_features'] = fc2_features[0]
    file_save_name = "/content/processed/" + str(index) +'preproc.pk' 
    index = index + 1   
    fid = open(file_save_name, 'wb')
    dump(data, fid)
    fid.close()

def show_images(X , width , hight , columns , rows):
  fig = plt.figure(figsize=(width , hight))
  for i in range(1, columns*rows +1):
      img = X[i-1]
      fig.add_subplot(rows, columns, i)
      plt.imshow(img)
  plt.show()



def test_images(path , shape , batch_size , model , feature_extract_model):
  # Image Data generator object
  train_datagen = ImageDataGenerator(rescale = 1/255 )
  # load images from directory
  train = train_datagen.flow_from_directory(path , target_size = shape ,
                                            batch_size = batch_size , class_mode = None)
  out_color   = []
  # loop through every image
  for img in tqdm(train[0]):
    # convert from rgb to LAB color space
    lab  = rgb2lab(img)[: , : , 0] # 224 , 224
    pred = np.zeros(shape + (3,))
    # first layer (gray scale)
    pred[: , : , 0] = lab 
    # gray scale with shape (224 , 224 , 3)
    lab = gray2rgb(lab)
    # lab with shape (1 , 224 , 224 , 3)
    lab = lab.reshape((1 , ) + lab.shape)
    # feature extraction from vgg (1 , 7 , 7 , 512)
    lab = feature_extract_model.predict(lab)
    # model decoder with shape (224 , 224 , 2)
    lab = model.predict(lab) * 128
    # the last two layer in the LAB img
    pred[: , : , 1:] = lab[0]
    # convert to rgb layer
    out_color.append(lab2rgb(pred))

  return np.array(out_color)


def data_generator_baseline(training_dir, num_train_samples, batch_size):
  current_batch_size = 0
      # loop through images for ever
  while 1:
    files = os.listdir(training_dir)
    for file_idx in range(num_train_samples): 
    # retrieve the photo feature
      if current_batch_size == 0:
        X1, Y = list() , list()
      file = training_dir + '/' + files[file_idx] # 1.pk
      fid = open(file, 'rb')
      data = load(fid)
      fid.close()
      features  = data['X_features'] 
      img_color = data['y_color_lab'] 
      X1.append(features)
      Y.append(img_color)
      current_batch_size += 1
      if current_batch_size == batch_size:
        current_batch_size = 0
        yield (np.array(X1), np.array(Y))
