import os
import csv
import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D, Activation, Lambda

'''
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    image = cv2.imread(source_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

input_shape = X_train[0].shape
print(input_shape)
'''
samples = []
with open('./data/driving_log.csv') as csvfile:
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
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

BATCH_SIZE = 32

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

ch, row, col = 3, 160, 320

def normalize(x):
    return x/127.5 - 1

model = Sequential()
model.add(Lambda(normalize, input_shape=(row, col, ch)))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)/BATCH_SIZE), 
                    validation_data=validation_generator, validation_steps=(len(validation_samples)/BATCH_SIZE), epochs=3)

model.save('model.h5')