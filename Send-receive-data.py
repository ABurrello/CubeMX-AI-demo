from keras.datasets import cifar10
import os
import serial
import sys
import numpy as np
import time
from keras.models import load_model
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

platform = 'subSystem-Linux'
dict_image = {'0': 'airplane', '1': 'automobile', '2': 'bird', '3': 'cat', '4': 'deer','5': 'dog', '6': 'frog', '7': 'horse', '8': 'ship', '9': 'truck'}
model = load_model('./saved_models/keras_cifar10_trained_model_200_epochs.h5')
if platform == 'subSystem-Linux':
	ser = serial.Serial('/dev/ttyS5', 115200)
elif platform == 'VirtualMachine':
	ser = serial.Serial('/dev/ttyACM0', 115200)
	import matplotlib.pyplot as plt
else:
	port = 'COM5'
	baud = 115200
	ser = serial.Serial(port, baud, timeout=0)
ser.flushInput()
ser.flushOutput()
for i in range(10,2000):
	## send image
	ser.write(x_test[i,:,:,:].flatten().tolist())
	time.sleep(.1)
	## reading result
	reading = ser.readline()
	pred_h7 = int.from_bytes(reading.split(b'\n')[0], "big")
	score = model.predict((x_test[i].reshape(1,32,32,3))/255, verbose=0)
	val = 0
	pred_python=0
	for j,pred in enumerate(score[0]):
		if pred>val:
			pred_python=j
			val = pred
	print('H7 Prediction = %d, Python Prediction = %d, Real Label = %d' % (pred_h7,pred_python,y_test[i][0]) )
	if platform == 'VirtualMachine':
		## Plot image
		plt.imshow(x_test[i])
		label = dict_image[str(pred_h7)]
		label_real = dict_image[str(y_test[i][0])]
		plt.title('Predicted label: '+ label + '\n Real label: ' + label_real)
		plt.show()



	##writing result
