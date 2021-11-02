

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics

import io
# import imageio
# from IPython.display import Image, display
# from ipywidgets import widgets, Layout, HBox
from PIL import Image

import math
from tensorflow.keras.utils import Sequence
import os
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


fpath = keras.utils.get_file(
    "moving_mnist.npy",
    "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
)
dataset = np.load(fpath)

# Swap the axes representing the number of frames and number of data samples.
dataset = np.swapaxes(dataset, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
dataset = dataset[:1000, ...]
# Add a channel dimension since the images are grayscale.
dataset = np.expand_dims(dataset, axis=-1)

# Split into train and validation sets using indexing to optimize memory.
indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)
train_index = indexes[: int(0.9 * dataset.shape[0])]
val_index = indexes[int(0.9 * dataset.shape[0]) :]
train_dataset = dataset[train_index]
val_dataset = dataset[val_index]

# Normalize the data to the 0-1 range.
train_dataset = train_dataset / 255
val_dataset = val_dataset / 255

# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y


# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))



# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(None, *x_train.shape[2:]))
# inp = layers.Input(shape=(None, 302,176,3))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=16,
    kernel_size=(3, 2),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=16,
    kernel_size=(3, 2),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=16,
    kernel_size=(3, 2),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
# x = layers.BatchNormalization()(x)
# x = layers.ConvLSTM2D(
#     filters=3,
#     kernel_size=(2, 1),
#     padding="same",
#     return_sequences=True,
#     activation="relu",
# )(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)


# Next, we will build the complete model and compile it.
model = keras.models.Model(inp, x)

# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-1,
#     decay_steps=10000,
#     decay_rate=0.9)


model.compile(
    # loss=keras.losses.binary_crossentropy,
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    metrics=[metrics.MeanAbsolutePercentageError(),metrics.MeanSquaredError()]

)

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
#3으로 바꿔보기
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2)

checkpoint_path = "chk.ckpt"

# 체크포인트 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
												period=20, # 1개의 epoch마다 저장
                                                 verbose=1)

# Define modifiable training hyperparameters.
epochs = 100
batch_size = 20




# Fit the model to the training data.
model.fit(
    x_train,
    y_train,
    batch_size=20,
    epochs=epochs,
    validation_data=(x_val,y_val),
    callbacks=[early_stopping, reduce_lr,cp_callback],
    verbose=1
)

model.save('mm.h5')