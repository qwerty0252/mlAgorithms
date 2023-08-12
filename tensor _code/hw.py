import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

#loadint the datasets
(X_train, y_train), (X_test, y_test) =  mnist.load_data()
X_train =  X_train.reshape(-1, 28*28).astype("float32") / 255.0
X_test =  X_test.reshape(-1, 28*28).astype("float32") / 255.0

#creating a model using the sequential API of keras
#very convient, not very flexible

model = keras.Sequential(
    [
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)
    ]
)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(X_test, y_test, batch_size=32, verbose=2)