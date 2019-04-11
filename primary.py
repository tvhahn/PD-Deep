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
# Parameters for CAE
cae_batch_size = 1024
cae_epochs = 200

# Parameters for FC layers
fc_batch_size = None
fc_epochs = 4000

x = 800
input_sig = Input(shape=(x, 1))
num_classes = 1

e_kernel_sizes = [3]  # encoder kernel size
e_filter_sizes = [64]
pool_sizes = [16]
dense_sizes = [20]
dense_layers = [2,3]
d_kernel_sizes = e_kernel_sizes
d_filter_sizes = e_filter_sizes
dropout_vals = [0.2,0.3,0.5,0]


for e_kernel_size in e_kernel_sizes:
    for e_filter_size in e_filter_sizes:
        for pool_size in pool_sizes:

            # create name so we can log model training. BN = batchnormalization
            NAME = "CAE_mean-BN-{}e_kern1-{}e_filt1-{}pool_size-{}d_kern-{}d_filt-{}".format(
                e_kernel_size,
                e_filter_size,
                pool_size,
                e_kernel_size,
                e_filter_size,
                int(time.time()),
            )
            SAVE_NAME = NAME

            def encoder(input_sig):
                # encoder
                # input = 800 x 1

                conv1 = Conv1D(
                    e_filter_size,
                    e_kernel_size,
                    activation="relu",
                    padding="same",
                    strides=1,
                )(input_sig)

                conv2 = Conv1D(
                    (e_filter_size*2),
                    e_kernel_size,
                    activation="relu",
                    padding="same",
                    strides=1,
                )(conv1)
                conv2 = BatchNormalization()(conv2)
                pool1 = MaxPooling1D(pool_size=pool_size)(conv2)

                conv3 = Conv1D(
                    (e_filter_size*4),
                    e_kernel_size,
                    activation="relu",
                    padding="same",
                    strides=1,
                )(pool1)

                conv3 = Conv1D(
                    (e_filter_size*4),
                    e_kernel_size,
                    activation="relu",
                    padding="same",
                    strides=1,
                )(conv3)
                conv3 = BatchNormalization()(conv3)

                return conv3

            def decoder(input_ec):
                # decoder

                conv4 = Conv1D(
                    (e_filter_size*4),
                    e_kernel_size,
                    activation="relu",
                    padding="same",
                    strides=1,
                )(input_ec)

                up2 = UpSampling1D(size=pool_size)(conv4)
                conv5 = Conv1D(
                    e_filter_size*2,
                    e_kernel_size,
                    activation="relu",
                    padding="same",
                    strides=1,
                )(up2)
                conv5 = BatchNormalization()(conv5)
                decoded = Conv1D(1, e_kernel_size, activation="relu", padding="same")(conv5)
                return decoded

            # Either load model, or train
            ### Uncomment to load ###
            # autoencoder = keras.models.load_model(
            #     "cae/CAE_mean-BN-3e_kern1-64e_filt1-16pool_size-3d_kern-64d_filt-1554563716.h5"
            # )

            autoencoder = Model(input_sig, decoder(encoder(input_sig)))
            autoencoder.compile(loss='mean_squared_error', optimizer ='adam')
            autoencoder.summary()

            tensorbard = TensorBoard(log_dir="logs/cae/{}".format(NAME))

            early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1,mode='min',
                                                   restore_best_weights=True)

            # Train Autoencoder if not loading a save ###
            autoencoder_train = autoencoder.fit(
                X_all_train,
                X_all_train,
                batch_size
                epochs=cae_epochs,
                verbose=0,
                validation_data=(X_all_val, X_all_val),
                callbacks=[tensorbard,early_stop],
                shuffle=True,
            )

            # # # Save model :https://www.tensorflow.org/tutorials/keras/save_and_restore_models
            # autoencoder.save("cae/{}.h5".format(SAVE_NAME))

            for dense_size in dense_sizes:
                for d_layer in dense_layers:
                    for dropout_val in dropout_vals:

                        def fc(enco):
                            den = GlobalAveragePooling1D()(enco)

                            for i in range(d_layer):
                                den = Dense(dense_size, activation="relu")(den)
                                den = Dropout(dropout_val)(den)

                            out = Dense(1, activation="sigmoid")(den)
                            return out

                        encode = encoder(input_sig)
                        full_model = Model(input_sig, fc(encode))

                        for l1, l2 in zip(
                            full_model.layers[0:8], autoencoder.layers[0:8]
                        ):
                            l1.set_weights(l2.get_weights())

                        for layer in full_model.layers[0:8]:
                            layer.trainable = False

                        print("#*#*#*#* - Layer 8:",full_model.layers[8])
                        full_model.compile(
                            loss="binary_crossentropy",
                            optimizer="adam",
                            metrics=["accuracy"],
                        )
                        param_size = full_model.count_params()
                        full_model.summary()

                        # Name: FALSE = encoder weights are fixed
                        NAME = "FC_mean_FALSE-{}p-{}e_kern-{}e_filt-{}pool_size-{}d_kern-{}d_filt-{}fc_units-{}fc_layers-{}dropout_val-{}".format(
                            param_size,
                            e_kernel_size,
                            e_filter_size,
                            pool_size,
                            e_kernel_size,
                            e_filter_size,
                            dense_size,
                            (d_layer),
                            dropout_val,
                            int(time.time()),
                        )

                        print(NAME)
                        print("############################################################################")
                        tensorbard = TensorBoard(log_dir="logs/fc4/{}".format(NAME))

                        path = "/home/tim/Documents/PD-Predictions/notebooks/fc_final"
                        os.mkdir(os.path.join(path, NAME))
                        path = os.path.join(path, NAME)

                        save_path = path + "/epoch.{epoch:02d}-val_acc.{val_acc:.2f}.h5"
                        save_path2 = (
                            path + "/epoch.{epoch:02d}-val_loss.{val_loss:.2f}.h5"
                        )
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

                        early_stop2 = EarlyStopping(monitor='val_loss', patience=600, verbose=1,mode='min') 

                        classify_train = full_model.fit(
                            X_train,
                            y_train,
                            epochs=fc_epochs,
                            verbose=0,
                            validation_data=(X_val, y_val),
                            callbacks=[tensorbard, checkpoint, checkpoint2, early_stop2],
                            shuffle=True,
                        )

                        full_model.save("fc/{}.h5".format(NAME))

