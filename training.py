# -*- coding: utf-8 -*-

# Data Copy
import tarfile
with tarfile.open('data.tar.xz') as f:
    f.extractall('.')

# Imports
import os
import cv2
import numpy as np
from tqdm import tqdm
from random import shuffle
import pickle
import matplotlib.pyplot as plt

import keras
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Global Variables
IMG_SIZE = 150
LR = 1e-3
batch_size = 32
num_classes = 47
epochs = 10
data_augmentation = False
num_predictions = 20
model_name = 'trained_model.h5'

# Data Preprocessing
path = "dataAll"

def create_train_data():
  training_data = []

  if os.path.exists("train_data.dat"):
    file = open('train_data.dat', 'rb')
    training_data = pickle.load(file)
    file.close()
    return training_data

  #img_count = 0
  for folder in tqdm(os.listdir(path)):
    p = path + "/" + folder
    files = os.listdir(p)
    for i in files:
      label = folder
      img_loc = p + "/" + i
      img = cv2.imread(img_loc,cv2.IMREAD_GRAYSCALE)
      img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
      training_data.append([np.array(img),np.array(label)])

  shuffle(training_data)

  file = open('train_data.dat', 'wb')
  pickle.dump(training_data, file)  
  file.close()

  return training_data

train_data = create_train_data()



### Data Split
train = train_data[:-6000]
test = train_data[-6000:]
# Training Data
x_train = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y_train = [i[1] for i in train]

y_train = to_categorical(y_train,47)
print(y_train)

# Testing Data
x_test = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)

y_test = [i[1] for i in test]
print(y_test)
y_test = to_categorical(y_test,47)
print(y_test)


# Design Model
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)


# Model Train
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Results
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Saving the model
model.save(model_name)