from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta

from math import sqrt

import tensorflow.compat.v1.keras as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Masking, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.framework import ops
from tensorflow.python.client import device_lib

from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.layers import Masking

from sklearn.metrics import mean_squared_error
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler

print(device_lib.list_local_devices())

# Data Parameter
n_steps = 365
n_test = 30
n_features = 1
n_future_pred = 180

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# split a univariate sequence into samples
def univariate_split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# create t feature with t+1 target
def s_to_super(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	target_label = df.iloc[:,-1]
	agg['target'] = target_label
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# create t feature with t+1 target
def produce_dataset(dataset):
	dataset = dataset.values
	reframed = s_to_super(dataset, 1, 1)
	values = reframed.values
	return values

# split X and Y and reshape to LSTM dataset
def multivariate_split_sequence(data):
	X, y = data[:, :-1], data[:, -1]
	print("Original Testing data shape X, y => ", X.shape, y.shape)
	X = X.reshape((X.shape[0], 1, X.shape[1]))
	print("** DATA SPLITTING COMPLETED **")
	print("Data shape X, y => ", X.shape, y.shape)
	return X, y

def generate_model(n_steps, n_features, data):
	RNN = tf.layers.LSTM
	# RNN = tf.layers.CuDNNLSTM
	model = Sequential()
	# model.add(RNN(50,  batch_input_shape=(training_data.shape[0], training_data.shape[1], training_data.shape[2])))
	model.add(RNN(50,  batch_input_shape=(data.shape[0], data.shape[1], data.shape[2])))
	model.add(Dense(1, activation="tanh"))
	model.compile(optimizer='adam', loss='mse')
	return model

def train_model(data):
    ### Model Training and Future Weather Prediction

	pred_actual = pd.DataFrame()
	future_pred_df = pd.DataFrame()
	rsme_score_df = pd.DataFrame()

	# Data Parameter
	n_steps = 365
	n_test = 30
	n_features = 1
	n_future_pred = 180

	# split into train/test samples
	X, y = univariate_split_sequence(data, n_steps)
	train_x = X[:-n_test]
	train_y = y[:-n_test]
	test_x = X[-n_test:]
	test_y = y[-n_test:]

	train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))
	test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], n_features))

	# create prediction model
	model = generate_model(n_steps, n_features, train_x)
	earlystopping = EarlyStopping(monitor='loss', patience=25, mode='min', restore_best_weights=True)
	callbacks = [earlystopping]

	# train the model
	history = model.fit(train_x, train_y, epochs=500, batch_size=train_x.shape[0], verbose=2, callbacks=callbacks)

	train_weights = model.get_weights()

	test_model = generate_model(n_steps, n_features, test_x)
	test_model.set_weights(train_weights)

	pred_y = test_model.predict(test_x, verbose=0)

	return train_x, train_y, test_x, test_y, pred_y, model, train_weights, history
