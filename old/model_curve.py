import csv
import pickle
import numpy as np
import math
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle	
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

# Fix error with TF and Keras
tf.python.control_flow_ops = tf
print('Modules loaded.')

# Load data
raw_data = None

root_path = r'C:\Users\MacNab\Develop\simulator-windows-64\train\\'

with open(root_path + 'driving_log.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	raw_data = list(reader)[1:]

# remove too slow data 	
raw_data = [line for line in raw_data if float(line[3]) > 0.07]
raw_data = shuffle(raw_data)
lines_train, lines_test = train_test_split(raw_data, test_size=0.2, random_state=0)
	
print('CSV file loaded.')	
print('Data Size: ', len(raw_data))
# data size: 19274
# raw_data is a 2-d array, 
# center,left,right,steering,throttle,brake,speed

image_array = np.asarray(Image.open(raw_data[0][0]))

height = len(image_array)
width = len(image_array[0])
print('Image Size: ', width, 'x', height)

steering = [float(line[3]) for line in raw_data]
steering_mean = sum(steering) / len(steering)
print('Steering Mean: ', steering_mean)

steering_res = [(float(line[3])-steering_mean)**2 for line in raw_data]
print('Steering MSE: ', sum(steering_res) / len(steering_res) )

# Model
with open("model.json", 'r') as jfile:
	model = model_from_json(jfile.read())
	model.compile('adam', 'mean_squared_error')

batch_size = 64

def generate_data(lines):

	batch_X, batch_y = [], []

	while 1:
		for i in range(0, len(lines)):
			
			X = np.asarray(Image.open(lines[i][0].strip()), dtype=np.uint8)
			y = float(lines[i][3])
			X_left = np.asarray(Image.open(lines[i][1].strip()), dtype=np.uint8)
			y_left = float(lines[i][3]) + 0.04
			X_right = np.asarray(Image.open(lines[i][2].strip()), dtype=np.uint8)
			y_right = float(lines[i][3]) - 0.04
			
			batch_X.append(X)
			batch_y.append(y)
			batch_X.append(X_left)
			batch_y.append(y_left)
			batch_X.append(X_right)
			batch_y.append(y_right) 
			
			if len(batch_y) >= batch_size:
				yield np.asarray(batch_X), np.asarray(batch_y)
				batch_X, batch_y = [], []
			
model.load_weights("model.h5")

history = model.fit_generator(generate_data(lines_train), samples_per_epoch=len(lines_train) * 3, nb_epoch=5, validation_data=generate_data(lines_test), nb_val_samples=len(lines_test) * 3)

# serialize weights to HDF5
model.save_weights("model_curve.h5")
print("Saved weights to disk")
