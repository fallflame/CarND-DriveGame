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

root_path = r'C:\Users\MacNab\Develop\simulator-windows-64\train\\'

with open(root_path + 'driving_log.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	raw_data = list(reader)[1:]

# remove too slow data 	
raw_data = [line for line in raw_data if float(line[6]) > 3]

# reinforce the curve 0.2
raw_data_curve = [line for line in raw_data if abs(float(line[3])) > 0.2]
for i in range(0, 5):
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
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(160, 320, 3))


x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dense(1)(x)

model = Model(input=base_model.input, output=x)

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
			y_right = float(lines[i][3]) - 0.05
			
			#renforce = 1
			#if abs(y) > 0.2 :
			#	renforce = 10
			
			#for j in range(0, renforce):
			batch_X.append(X)
			batch_y.append(y)
			batch_X.append(X_left)
			batch_y.append(y_left)
			batch_X.append(X_right)
			batch_y.append(y_right) 
			
			if len(batch_y) >= batch_size:
				yield np.asarray(batch_X), np.asarray(batch_y)
				batch_X, batch_y = [], []

for i, layer in enumerate(model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:19]:
   layer.trainable = False
for layer in model.layers[19:]:
   layer.trainable = True

# Compile and train the model
model.compile('adam', 'mean_squared_error')

raw_data = shuffle(raw_data)
lines_train, lines_test = train_test_split(raw_data, test_size=0.2, random_state=0)
history = model.fit_generator(generate_data(lines_train), samples_per_epoch=len(lines_train) * 3, nb_epoch=40, validation_data=generate_data(lines_test), nb_val_samples=len(lines_test) * 3)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
