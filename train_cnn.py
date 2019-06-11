import keras
from keras.datasets import mnist
from scipy.misc import imread, imresize
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np


batch_size = 128
num_classes = 10
epochs = 100
img_rows, img_cols = 28, 28

train_meta_file = open("label_processed_digits.txt", mode="r")
test_meta_file = open("test_y.txt", mode="r")
train_meta = train_meta_file.readlines()
test_meta = test_meta_file.readlines()

path_files = []
lb = []

for label in train_meta:
	path_files.append(label.split(", ")[0])
	lb.append(label.split(", ")[1].replace("\n", ""))

x_train = []

for path_file in path_files:
	x = imread(path_file,mode='L')
	# x = x.reshape(1,28,28,1)
	# x = x.astype('float32')
	x_train.append(x)

y_train = to_categorical(lb)

path_files = []
lb = []

for label in test_meta:
	path_files.append(label.split(", ")[0])
	lb.append(label.split(", ")[1].replace("\n", ""))

x_test = []

for path_file in path_files:
	x = imread(path_file,mode='L')
	# x = x.reshape(1,28,28,1)
	# x = x.astype('float32')
	x_test.append(x)

y_test = to_categorical(lb)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = load_model('cnn.h5')
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(decay=0.1),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('cnn_x.h5')