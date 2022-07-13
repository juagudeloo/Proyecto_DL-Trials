import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, GlobalMaxPool1D, Dense, Dropout
from tensorflow.keras.models import Model

# Some toy data
train_x = np.random.normal(size=(7200, 40))
train_y = np.random.choice([0,1,2], size=(7200))

dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y))
dataset = dataset.batch(256)

#### - Define your model here - ####
model = tf.keras.Sequential()
model.add(Conv1D(20, 2, activation='relu', input_shape=(40,)))
model.add(Conv1D(10, 2, activation='relu'))
model.add(Dense(1, name="output"))
lr = 0.001
opt = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.metrics.MeanSquaredError()
model.compile(optimizer = opt, loss = loss, metrics = loss)
print("model compiled!")
model.summary()
history = model.fit(dataset, epochs=EPOCHS)
