import numpy as np
import tensorflow as tf

# Some toy data
train_x = np.random.normal(size=(7200, 40))
train_y = np.random.choice([0,1,2], size=(7200))

dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y))
dataset = dataset.batch(256)

#### - Define your model here - ####

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
history = model.fit(dataset, epochs=EPOCHS)
