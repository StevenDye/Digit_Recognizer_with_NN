import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

img_rows, img_cols = 28, 28
num_classes = 10

def data_prep(raw):
	out_y = keras.utils.to_categorical(raw.label, num_classes)

	# raw.label looks like this: 	0		1
	#								1		0
	#								2		1
	#										..
	#								41998	6
	#								41999	9

	num_images = raw.shape[0] # num_images is 42000
	x_as_array = raw.values[:,1:] # Gives data as Numpy array. It's a big array
	x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1) # reshape to 4D array
	out_x = x_shaped_array / 255 # Normalizes. Gray scale ranges from 0-255
	return out_x, out_y

def data_test(raw):

	num_images = raw.shape[0]
	x_as_array = raw.values[:,0:] # Gives data as Numpy array
	x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1) # reshape to 4D array
	out_x = x_shaped_array / 255
	return out_x

train_file = 'input/train.csv'
raw_data = pd.read_csv(train_file)

x, y = data_prep(raw_data) # Both have length of 42000

model = Sequential()
model.add(Conv2D(20, kernel_size=(3,3),
				activation='relu',
				input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax')) # Output

model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer='adam',
				metrics=['accuracy'])

model.fit(x, y, # x is predictor, y is target
			batch_size=128,
			epochs=2,
			validation_split = 0.2)

# PREDICTIONS
test_file = 'input/test.csv'
test_data = pd.read_csv(test_file)
#test_data.reshape([None,28000,28,28])

x_test = data_test(test_data)
predictions = model.predict(x_test, verbose=0) # input data needs to be a numpy array
#print('The predictions are: {}'.format(predictions))

# OUTPUT
df = pd.DataFrame(predictions)
df.index+=1
df['Label'] = df.idxmax(axis=1)
df_final = df['Label']
df_final.index.name='ImageId'
df_final.to_csv('output/results_NN.csv', header=True)