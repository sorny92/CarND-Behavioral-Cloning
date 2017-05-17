import os
import csv
import cv2
import numpy as np
import sklearn

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D, Activation, Lambda, Cropping2D, GaussianNoise
from keras.preprocessing.image import ImageDataGenerator


samples = []
#data_low > data > data_contreras > data_low_v2
with open('./data_long/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data_long/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                image_flipped = np.fliplr(center_image)
                image_flipped_angle = -center_angle
                images.append(image_flipped)
                angles.append(image_flipped_angle)
                images.append(center_image)
                angles.append(center_angle)

            # trim image t o only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

BATCH_SIZE = 2
EPOCHS = 2

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

ch, row, col = 3, 160, 320

def preprocess(x):
    x = x/127.5 - 1
    return x
    
model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(row, col, ch)))
model.add(Lambda(preprocess))

model.add(Conv2D(24, (5, 5)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(36, (5, 5)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)/BATCH_SIZE), 
                    validation_data=validation_generator, validation_steps=(len(validation_samples)/BATCH_SIZE), epochs=EPOCHS)

model.save('model_def_hard.h5')
