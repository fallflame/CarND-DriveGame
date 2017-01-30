import csv
import pickle
import numpy as np
import math
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import model_from_json


# Fix error with TF and Keras
tf.python.control_flow_ops = tf
print('Modules loaded.')

# Load data
raw_data = None

root_path = r'C:\Users\MacNab\Develop\simulator-windows-64\train\\'

with open(root_path + 'driving_log.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	raw_data = list(reader)[1:]
	
# data size: 19274
# raw_data is a 2-d array, 
# center,left,right,steering,throttle,brake,speed

with open("model.json", 'r') as jfile:
	model = model_from_json(jfile.read())
model.compile('adam', 'mean_squared_error')
model.load_weights("model.h5")


center_sum = 0
left_sum = 0
right_sum = 0

for j in range(0, len(raw_data)):
	for i in range(0, 3):
		image_array = np.asarray(Image.open(raw_data[j][i].strip()))

		image_array = image_array[None, :, :, :]
		
		steering_angle = float(model.predict(image_array, batch_size=1))
		
		if (i == 0) :
			center_sum += steering_angle
		elif (i == 1) :
			left_sum += steering_angle
		else :
			right_sum += steering_angle
		
print("center: ", center_sum / len(raw_data))
print("left: ", left_sum / len(raw_data))
print("right: ", right_sum / len(raw_data))



# TODO: Compile and train the model


