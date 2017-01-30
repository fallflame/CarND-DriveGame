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

root_path = r'C:\Users\MacNab\Develop\simulator-windows-64\data\\'

with open(root_path + 'driving_log.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	raw_data = list(reader)[1:]

# remove too slow data 	
raw_data = [line for line in raw_data if float(line[6]) > 3]

# Match the image with next steering
import datetime
def read_time(filename):
	index = filename.index('center')
	year = filename[index+7: index+11]
	month = filename[index+12: index+14]
	day = filename[index+15: index+17]
	hour = filename[index+18: index+20]
	minuit = filename[index+21: index+23]
	sec = filename[index+24: index+26]
	milsec = filename[index+27: index+30]
	return datetime.datetime(int(year),int(month),int(day), int(hour), int(minuit), int(sec), int(milsec))

for i in range(0, len(raw_data)-1):
	ts_1 = read_time(raw_data[i][0])
	ts_2 = read_time(raw_data[i+1][0])
	if ts_2.timestamp() - ts_1.timestamp() < 1 :
		raw_data[i][3] = (float(raw_data[i][3]) + float(raw_data[i+1][3])) / 2

# reinforce the curve 0.2

raw_data_curve = [line for line in raw_data if abs(float(line[3])) > 0.2]
for i in range(5):
	raw_data.extend(raw_data_curve)

	
print('CSV file loaded.')	
print('Data Size: ', len(raw_data))
# data size: 19274
# raw_data is a 2-d array, 
# center,left,right,steering,throttle,brake,speed

##################
#    meta data   #
##################

image_array = np.asarray(Image.open(raw_data[0][0]))

height = len(image_array)
width = len(image_array[0])
print('Image Size: ', width, 'x', height)

steering = [float(line[3]) for line in raw_data]
steering_mean = sum(steering) / len(steering)
print('Steering Mean: ', steering_mean)

steering_res = [(float(line[3])-steering_mean)**2 for line in raw_data]
print('Steering MSE: ', sum(steering_res) / len(steering_res) )


##############
#    Model   #
##############
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

model = Sequential()

model.add(BatchNormalization(input_shape=(height, width, 3)))

# 160 x 320 x 3
model.add(Convolution2D(40, 11, 11, subsample = (2, 2)))
# 75 x 155 x 40
model.add(Activation('relu'))
model.add(MaxPooling2D())
# 37 x 77 x 40

model.add(Convolution2D(60, 7, 7, subsample = (2, 2)))
# 16 x 36 x 60
model.add(Activation('relu'))
model.add(MaxPooling2D())
# 8 x 18 x 60

model.add(Convolution2D(80, 3, 3))
# 6 x 16 x 80
model.add(Activation('relu'))

model.add(Convolution2D(110, 3, 1))
# 4 * 16 * 110
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(0.5))
# 7040

model.add(Dense(1000))
model.add(Activation('relu'))

model.add(Dense(200))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

##############
#  Training  #
##############
batch_size = 64

def generate_data(lines):

	batch_X, batch_y = [], []

	while 1:
		for i in range(0, len(lines)):
			
			X = np.asarray(Image.open(lines[i][0].strip()), dtype=np.uint8)
			y = float(lines[i][3])
			X_left = np.asarray(Image.open(lines[i][1].strip()), dtype=np.uint8)
			y_left = float(lines[i][3]) + 0.05
			X_right = np.asarray(Image.open(lines[i][2].strip()), dtype=np.uint8)
			y_right = float(lines[i][3]) - 0.045
			
			batch_X.append(X)
			batch_y.append(y)
			batch_X.append(X_left)
			batch_y.append(y_left)
			batch_X.append(X_right)
			batch_y.append(y_right) 
			
			if len(batch_y) >= batch_size:
				yield np.asarray(batch_X), np.asarray(batch_y)
				batch_X, batch_y = [], []
				
# Compile and train the model
model.compile('adam', 'mean_squared_error')
model.load_weights("model.h5")

for layer in model.layers:
	print (layer, layer.output_shape)

raw_data = shuffle(raw_data)
lines_train, lines_test = train_test_split(raw_data, test_size=0.2, random_state=0)
history = model.fit_generator(generate_data(lines_train), samples_per_epoch=len(lines_train) * 3, nb_epoch=1, validation_data=generate_data(lines_test), nb_val_samples=len(lines_test) * 3)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
