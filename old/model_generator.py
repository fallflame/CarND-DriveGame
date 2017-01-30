import csv
import pickle
import numpy as np
import math
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Fix error with TF and Keras
tf.python.control_flow_ops = tf
print('Modules loaded.')

# Load data
raw_data = None

with open(r'C:\Users\MacNab\Develop\simulator-windows-64\train\driving_log.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	raw_data = list(reader)[1:]

print('CSV file loaded.')	
print('Data Size: ', len(raw_data))
# data size: 19274
# raw_data is a 2-d array, 
# center,left,right,steering,throttle,brake,speed

image_array = np.asarray(Image.open(raw_data[0][0]))

height = len(image_array)
width = len(image_array[0])
print('Image Size: ', width, 'x', height)

# Model	
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(BatchNormalization(input_shape=(height, width, 3)))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(20, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(1))

# TODO: Compile and train the model
model.compile('adam', 'mse')

def load_data():
	while 1:
		for i in range(0, len(raw_data)):
			im = Image.open(raw_data[i][0])
			yield (np.asarray(im, dtype=np.uint8), raw_data[i][3])

history = model.fit_generator(load_data(), samples_per_epoch=1000, nb_epoch=10)

def load_data():
	while 1:
		for i in range(0, len(raw_data)):
			im = Image.open(raw_data[i][1])
			yield (np.asarray(im, dtype=np.uint8), raw_data[i][3] + 0.2)

history = model.fit_generator(load_data(), samples_per_epoch=1000, nb_epoch=10)

def load_data():
	while 1:
		for i in range(0, len(raw_data)):
			im = Image.open(raw_data[i][1])
			yield (np.asarray(im, dtype=np.uint8), raw_data[i][3] - 0.2)

history = model.fit_generator(load_data(), samples_per_epoch=1000, nb_epoch=10)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

"""
# TODO: Preprocess data & one-hot encode the labels
X_normalized_test = normalize_grayscale(X_test)
y_one_hot_test = label_binarizer.fit_transform(y_test)

# TODO: Evaluate model on test data
metrics = model.evaluate(X_normalized_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))
	
"""