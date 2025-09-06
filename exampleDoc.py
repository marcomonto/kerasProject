from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from typing import Tuple

# Type annotations for data variables
X: NDArray[np.float64]
y: NDArray[np.int32]
X_train: NDArray[np.float64]
X_test: NDArray[np.float64]
y_train: NDArray[np.int32]
y_test: NDArray[np.int32]

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.20)
print("Training set dimensions (train_data):")
print(X_train.shape)

# Type annotations for model variables
model: tf.keras.Sequential
history: tf.keras.callbacks.History
test_loss: float
test_pr: float

model = models.Sequential()
#The first layer that you define is the input layer. This
#layer needs to know the input dimensions of your data.
# Dense = fully connected layer (each neuron is fully
#connected to all neurons in the previous layer)
model.add(layers.Dense(64, activation='relu',
input_shape=(X_train.shape[1],)))
# Add one hidden layer (after the first layer, you don't need
#to specify the size of the input anymore)

model.add(layers.Dense(64, activation='relu'))
# If you don't specify anything, no activation is applied (ie.
#"linear" activation: a(x) = x)
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
metrics=[tf.keras.metrics.Precision()])
# Fit the model to the training data and record events into a
#History object.
history = model.fit(X_train, y_train, epochs=10, batch_size=1,
validation_split=0.2, verbose=1)
# Model evaluation
test_loss, test_pr = model.evaluate(X_test, y_test)
print(test_pr)


# Plot loss (y axis) and epochs (x axis) for training set and
#validation set
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history.epoch,
np.array(history.history['loss']),label='Train loss')
plt.plot(history.epoch,
np.array(history.history['val_loss']),label = 'Val loss')
plt.legend()
plt.savefig('training_loss.png')
print("Grafico salvato come 'training_loss.png'")