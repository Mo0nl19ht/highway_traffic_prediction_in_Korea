

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


class Dataloader(Sequence):

    def __init__(self, data_list):
        self.data_list=data_list
        # self.batch_size = batch_size
    
    def __len__(self):
        return math.ceil(len(self.data_list))

		# batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):
				# sampler의 역할(index를 batch_size만큼 sampling해줌)
        x_path="D:/npz_gray_7/batch/x/"
        y_path="D:/npz_gray_7/batch/y/"
        # print(f"{idx} idx 번호")
        # print(f"{self.data_list[idx]} x 번호")
        return np.load(f"{x_path}{self.data_list[idx]}.npz")['x'] , np.load(f"{y_path}{self.data_list[idx]}.npz")['y']

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import random

num=60
batch_size=16

k=int(num/20)

print('hello world')
val_index=random.choices(range(num),k=k)
train_index=[]
for i in list(range(num)):
    if i not in val_index:
        train_index.append(i)

train_index=list(set(train_index))
val_index=list(set(val_index))
print("val_index")
print(val_index)


train_loader = Dataloader(train_index)

valid_loader = Dataloader(val_index)


# Construct the input layer with no definite frame size.
# inp = layers.Input(shape=(None, *x_train.shape[2:]))
inp = layers.Input(shape=(None,302,176,1))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=3,
    kernel_size=(3, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=3,
    kernel_size=(3, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=3,
    kernel_size=(3, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
# x = layers.BatchNormalization()(x)
# x = layers.ConvLSTM2D(
#     filters=3,
#     kernel_size=(3, 1),
#     padding="same",
#     return_sequences=True,
#     activation="relu",
# )(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3,1), activation="sigmoid", padding="same"
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
    optimizer=keras.optimizers.Adam(learning_rate=0.008),
    metrics=[metrics.MeanAbsolutePercentageError(),metrics.MeanSquaredError()]

)


# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
#3으로 바꿔보기
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2)

checkpoint_path = "chk_gray.ckpt"

# 체크포인트 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
												period=20, # 1개의 epoch마다 저장
                                                 verbose=1)

# Define modifiable training hyperparameters.
epochs = 100
# batch_size = 8




# Fit the model to the training data.
model.fit(
    train_loader,
    # batch_size=batch_size,
    epochs=epochs,
    validation_data=valid_loader,
    callbacks=[early_stopping, reduce_lr,cp_callback],
    verbose=1
)

model.save('my_model_mse_1026_gray_batch16_win14.h5')
