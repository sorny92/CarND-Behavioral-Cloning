import os
import csv
import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D, Activation, Lambda, Cropping2D

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
with open('./data_v2/driving_log.csv') as csvfile:
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
                name = './data_v2/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
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

BATCH_SIZE = 4
EPOCHS = 2

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

ch, row, col = 3, 160, 320

def normalize(x):
    return x/127.5 - 1

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch)))
model.add(Lambda(normalize))
model.add(Conv2D(15, (2, 2)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(36, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))
'''model.add(Conv2D(10, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Conv2D(12, (4, 4)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(60))
model.add(Activation('relu'))aw
model.add(Dense(90))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))'''

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)/BATCH_SIZE), 
                    validation_data=validation_generator, validation_steps=(len(validation_samples)/BATCH_SIZE), epochs=EPOCHS)

model.save('model.h5')