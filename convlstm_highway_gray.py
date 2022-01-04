

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


folder_name="npz_gray_kang"
#npz_x의 갯수
num=31
batch_size=32
#(24,14,1) fix
#(24,31,1) kang
# (24,30,1) olym
img_shape=(24,31,1)


class Dataloader(Sequence):

    def __init__(self, data_list):
        self.data_list=data_list
        # self.batch_size = batch_size
    
    def __len__(self):
        return math.ceil(len(self.data_list))

		# batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):
				# sampler의 역할(index를 batch_size만큼 sampling해줌)
        x_path=f"../{folder_name}/batch/x/"
        y_path=f"../{folder_name}/batch/y/"
        return np.load(f"{x_path}{self.data_list[idx]}.npz")['x'] , np.load(f"{y_path}{self.data_list[idx]}.npz")['y']

from tensorflow.python.client import device_lib



import random


# k=int(num/20)


# val_index=random.choices(range(num),k=k)
val_index=[3]
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


inp = layers.Input(shape=(None,img_shape[0],img_shape[1],img_shape[2]))


# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.


filter_size=64
drop=0.1
k_size=(3,3)

lr=0.0003
loss_f="mape"


x = layers.ConvLSTM2D(
    filters=filter_size,
    kernel_size=k_size,
    padding="same",
    return_sequences=True,
    activation="relu",
    dropout=drop
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=filter_size,
    kernel_size=k_size,
    padding="same",
    return_sequences=True,
    activation="relu",
    dropout=drop
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=filter_size,
    kernel_size=k_size,
    padding="same",
    return_sequences=True,
    activation="relu",
    dropout=drop
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=filter_size,
    kernel_size=k_size,
    padding="same",
    return_sequences=True,
    activation="relu",
    dropout=drop
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=filter_size,
    kernel_size=k_size,
    padding="same",
    return_sequences=True,
    activation="relu",
    dropout=drop
)(x)
# x = layers.BatchNormalization()(x)
# x = layers.ConvLSTM2D(
#     filters=filter_size,
#     kernel_size=(3, 2),
#     padding="same",
#     return_sequences=True,
#     activation="relu",
#     dropout=0.1
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

def MAPE(y_test, y_pred):
    # print(y_test.shape, y_pred.shape)
    y_t=tf.where(tf.math.equal(y_test, 0),1e-17,y_test)
    y_p=tf.where(tf.math.equal(y_test, 0),1e-17,y_pred)
    

    return keras.losses.MAPE(y_t,y_p)

model.compile(
    # loss=keras.losses.binary_crossentropy,
    # loss=keras.losses.MeanSquaredError(),
    # loss='binary_crossentropy',
    loss=MAPE,
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    metrics=[MAPE,metrics.MeanSquaredError()]

)

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
#3으로 바꿔보기
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

checkpoint_path = "chk_gray.ckpt"

# 체크포인트 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
												period=20, # 1개의 epoch마다 저장
                                                 verbose=1)

# Define modifiable training hyperparameters.
epochs = 300
# batch_size = 8


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="log", histogram_freq=1)


print(model.summary())

# Fit the model to the training data.
model.fit(
    train_loader,
    # batch_size=batch_size,
    epochs=epochs,
    validation_data=valid_loader,
    callbacks=[early_stopping, reduce_lr,cp_callback,tensorboard_callback],
    # callbacks=[reduce_lr,cp_callback,tensorboard_callback],
    verbose=1
)

model.save(f'kang_1223/KangByeon_{filter_size}_lay5_{lr}_{loss_f}_{drop}_{k_size}.h5')
