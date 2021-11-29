import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import itertools
import random
import warnings
import os
import numpy as np
import cv2
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
warnings.simplefilter(action='ignore', category=FutureWarning)


def fun2(word_dict):
    print("training started")
    train_path = r'C:\Capstone_dev_ops\code\train'
    test_path = r'C:\Capstone_dev_ops\code\test'

    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(64,64), class_mode='categorical', batch_size=10,shuffle=True)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(64,64), class_mode='categorical', batch_size=10, shuffle=True)

    imgs, labels = next(train_batches)

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())

    model.add(Dense(64,activation ="relu"))
    model.add(Dense(128,activation ="relu"))
    #model.add(Dropout(0.2))
    model.add(Dense(128,activation ="relu"))
    #model.add(Dropout(0.3))
    model.add(Dense(5,activation ="softmax"))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


    model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

    history2 = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop],  validation_data = test_batches)#, checkpoint])
    imgs, labels = next(train_batches) # For getting next batch of imgs...

    imgs, labels = next(test_batches) # For getting next batch of imgs...
    scores = model.evaluate(imgs, labels, verbose=0)
    print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

    model.save('best_model_3.h5')

    print(history2.history)

    imgs, labels = next(test_batches)

    model = keras.models.load_model(r"best_model_3.h5")

    scores = model.evaluate(imgs, labels, verbose=0)
    print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

    model.summary()

    scores #[loss, accuracy] on test data...
    model.metrics_names

    #word_dict = {0:'I',1:'love',2:'food',3:'hot'}

    predictions = model.predict(imgs, verbose=0)
    print("predictions on a small set of test data--")
    print("")
    for ind, i in enumerate(predictions):
        print(word_dict[np.argmax(i)], end='   ')

    print('Actual labels')
    for i in labels:
        print(word_dict[np.argmax(i)], end='   ')

    #print(imgs.shape)

    history2.history

    print("#############","done","###############")
