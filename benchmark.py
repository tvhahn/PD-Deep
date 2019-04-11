# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.models import Model, Sequential
from keras.optimizers import Adadelta, RMSprop, SGD, Adam
from keras.layers import (
    Input,
    Conv1D,
    GlobalAveragePooling1D,
    MaxPooling1D,
    Flatten,
    Activation,
    UpSampling1D,
    AveragePooling1D,
    Reshape,
)
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy, binary_crossentropy


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Import data and normalize


def normalizer(a, min_val, max_val):
    # min-max scaling if wanted
    # https://codereview.stackexchange.com/questions/185785/scale-numpy-array-to-certain-range
    col, row = np.shape(a)
    for i in range(col):
        a[i] = np.interp(a[i], (min_val, max_val), (0, 1))
    return a


# Load Pickles

# all_train
pickle_in = open("all_train_mean_NOSCALE.pickle", "rb")
X_all = pickle.load(pickle_in)

# df_extra1
pickle_in = open("df_extra1_mean_NOSCALE.pickle", "rb")
X_extra1 = pickle.load(pickle_in)

# df_extra2
pickle_in = open("df_extra2_mean_NOSCALE.pickle", "rb")
X_extra2 = pickle.load(pickle_in)

# df_extra3
pickle_in = open("df_extra3_mean_NOSCALE.pickle", "rb")
X_extra3 = pickle.load(pickle_in)

# X_train
pickle_in = open("X_train.pickle", "rb")
X_train = pickle.load(pickle_in)

# X_val
pickle_in = open("X_val.pickle", "rb")
X_val = pickle.load(pickle_in)

# X_test
pickle_in = open("X_test.pickle", "rb")
X_test = pickle.load(pickle_in)

# y_train
pickle_in = open("y_train.pickle", "rb")
y_train = pickle.load(pickle_in)

# y_val
pickle_in = open("y_val.pickle", "rb")
y_val = pickle.load(pickle_in)

# y_test
pickle_in = open("y_test.pickle", "rb")
y_test = pickle.load(pickle_in)

X_all = np.concatenate((X_all, X_extra1, X_extra2, X_extra3), axis=0)

# If needed, normalize the data between 0-1
min_all = X_all.min()
max_all = X_all.max()

X_all = normalizer(X_all, min_all, max_all)
X_train = normalizer(X_train, min_all, max_all)
X_val = normalizer(X_val, min_all, max_all)
X_test = normalizer(X_test, min_all, max_all)

# Reshape
X_all = X_all.reshape([-1, 800, 1]).astype("float32")
X_train = X_train.reshape([-1, 800, 1]).astype("float32")
X_val = X_val.reshape([-1, 800, 1]).astype("float32")
X_test = X_test.reshape([-1, 800, 1]).astype("float32")

# split X_all data
X_all_train, X_all_val, y_junk, y_junk = train_test_split(
    X_all, X_all, test_size=0.2, random_state=42
)

#### Hyperparameter Sweep ###

fc_batch_size = None
fc_epochs = 7000

x = 800
input_sig = Input(shape=(x, 1))
num_classes = 1

kernel_sizes = [5,3]  # encoder kernel size
filter_sizes = [64,20,30]
kernel_sizes2 = [5,3]
filter_sizes2 = [64,128,256]
conv_layers = [1]
pool_sizes = [16]
dense_sizes = [12,20]
dense_layers = [1,2,3]
dropout_vals = [0,0.2,0.5]


for kernel_size in kernel_sizes:
    for filter_size in filter_sizes:
        for pool_size in pool_sizes:
            for kernel_size2 in kernel_sizes2:
                for filter_size2 in filter_sizes2:
                    for conv_layer in conv_layers:
                        for dense_size in dense_sizes:
                            for d_layer in dense_layers:
                                for dropout_val in dropout_vals:

                                    model = Sequential()
                                    model.add(
                                        Conv1D(
                                            filter_size,
                                            kernel_size,
                                            activation="relu",
                                            input_shape=(800, 1),
                                        )
                                    )

                                    for j in range(conv_layer):
                                        model.add(MaxPooling1D(pool_size))
                                        model.add(
                                            Conv1D(
                                                filter_size2,
                                                kernel_size2,
                                                strides=1,
                                                activation="relu",
                                            )
                                        )

                                    model.add(GlobalAveragePooling1D())

                                    for i in range(d_layer):

                                        model.add(Dense(dense_size, activation="relu"))
                                        model.add(Dropout(dropout_val))

                                    model.add(Dense(1, activation="sigmoid"))

                                    model.compile(
                                        loss="binary_crossentropy",
                                        optimizer="adam",
                                        metrics=["accuracy"],
                                    )
                                    param_size = model.count_params()
                                    # model.summary()

                                    NAME = "CNN-mean-p{}-{}kern-{}filt-{}conv_layer-{}kernel2-{}filter2-{}pool_size-{}fc_units-{}fc_layers-{}dropout_val-{}".format(
                                        param_size,
                                        kernel_size,
                                        filter_size,
                                        conv_layer,
                                        kernel_size2,
                                        filter_size2,
                                        pool_size,
                                        dense_size,
                                        d_layer,
                                        dropout_val,
                                        int(time.time()),
                                    )

                                    # Create a sub-folder for the logs if needed
                                    print(NAME)
                                    print("#################################################")
                                    tensorbard = TensorBoard(
                                        log_dir="logs/cnn_simple2/{}".format(NAME)
                                    )

                                    path = "/home/tim/Documents/PD-Predictions/notebooks/cnn_simple2"
                                    os.mkdir(os.path.join(path, NAME))
                                    path = os.path.join(path, NAME)

                                    # save paths for the best models based on val-loss and val-accuracy
                                    save_path = (
                                        path
                                        + "/epoch.{epoch:02d}-val_acc.{val_acc:.2f}.h5"
                                    )
                                    save_path2 = (
                                        path
                                        + "/epoch.{epoch:02d}-val_loss.{val_loss:.2f}.h5"
                                    )

                                    # stop on validation loss
                                    early_stop2 = EarlyStopping(monitor='val_loss', patience=800, verbose=1,mode='min')

                                    checkpoint = ModelCheckpoint(
                                        filepath=save_path,
                                        monitor="val_acc",
                                        verbose=0,
                                        save_best_only=True,
                                        mode="max",
                                    )
                                    checkpoint2 = ModelCheckpoint(
                                        filepath=save_path2,
                                        monitor="val_loss",
                                        verbose=0,
                                        save_best_only=True,
                                        mode="min",
                                    )

                                    # train the model
                                    classify_train = model.fit(
                                        X_train,
                                        y_train,
                                        epochs=fc_epochs,
                                        verbose=0,
                                        shuffle=True,
                                        validation_data=(X_val, y_val),
                                        callbacks=[tensorbard, checkpoint, checkpoint2, early_stop2],
                                    )

