import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel("ERROR")
import numpy as np
from keras import metrics
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LeakyReLU, Conv2D, MaxPool2D, UpSampling2D, Input, Lambda, BatchNormalization, Activation, Dense, AveragePooling2D, MaxPooling2D, Flatten, Dropout, Concatenate
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
import time
import pandas as pd
import gc
from tensorflow.keras import backend as k
from tqdm import tqdm
import math
from tensorflow.keras.callbacks import Callback
from sklearn import preprocessing


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()

def plot_history(history, path):
    """
    Plot keras training history
    :param history: keras history
    :return: None
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model acc')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')
    
    plt.savefig(path)

np.random.seed(1)
tf.random.set_seed(1)


def MAPE_metric(y_true, y_pred):
    return k.mean(k.abs((y_true - y_pred) / y_true)) * 100


def pool_top_k(x, k):
    tk = tf.sort(x, axis=0, direction='DESCENDING')[:, :k, :]
    ptk = tf.reduce_mean(tk, axis=1, keepdims=True)
    return ptk
def top_k(x, k):
    tk = tf.sort(x, axis=0, direction='DESCENDING')[:, :k, :]
    return tk


def construct_model(fold_idx, x_train, covar_train, k):
    print('Model input shape :: ', x_train[0].shape)

    cell_ipt = Input(shape=x_train[0].shape)
    covar_ipt = Input(shape=covar_train[0].shape)

    x = Conv2D(3, kernel_size=(1, 23),activation=None, name='cv1', kernel_initializer='he_normal')(cell_ipt)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = LeakyReLU()(x)

    x = Conv2D(3, kernel_size=(1, 1),activation=None, name='cv2', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = LeakyReLU()(x)


    # x = Dropout(0.5)(x)
    x = AveragePooling2D(pool_size=(x_train.shape[1], 1))(x)
    x = Flatten()(x)

    y = Dense(3, activation='relu')(covar_ipt)
    y = Dense(3, activation='relu')(y)
    combined = Concatenate()([x, y])


    z = Dense(10, activation=None)(combined)
    z = Activation("relu")(z)

    z = Dense(1, activation=None)(z)
    out = Activation("linear")(z)

    prediction_model = Model([cell_ipt, covar_ipt], out)
    print(prediction_model.summary())
    return prediction_model

accs = []
print('Running')
for fidx in range(0, 5):
# for fidx in range(4, -1, -1):
    # if fidx != 0: continue
    print('===============Training Fold {}==============='.format(fidx))
    # X_train = np.load('/mnt/Dev_ssd/Dev_ssd/hiv_brain/dl/data_folds_raw/X_train_{}_arsinh.npy'.format(fidx))
    # X_val = np.load('/mnt/Dev_ssd/Dev_ssd/hiv_brain/dl/data_folds_raw/X_val_{}_arsinh.npy'.format(fidx))
    
    X_train = np.load('/mnt/Dev_ssd/Dev_ssd/hiv_brain/dl/data_folds_raw/X_train_{}_norm.npy'.format(fidx))
    X_val = np.load('/mnt/Dev_ssd/Dev_ssd/hiv_brain/dl/data_folds_raw/X_val_{}_norm.npy'.format(fidx))
    

    y_train = np.load('/mnt/Dev_ssd/Dev_ssd/hiv_brain/dl/data_folds_raw/y_train_{}.npy'.format(fidx))
    y_val = np.load('/mnt/Dev_ssd/Dev_ssd/hiv_brain/dl/data_folds_raw/y_val_{}.npy'.format(fidx))
    score_train = np.load('/mnt/Dev_ssd/Dev_ssd/hiv_brain/dl/data_folds_raw/score_train_{}.npy'.format(fidx))
    score_val = np.load('/mnt/Dev_ssd/Dev_ssd/hiv_brain/dl/data_folds_raw/score_val_{}.npy'.format(fidx))
    covar_train = np.load('/mnt/Dev_ssd/Dev_ssd/hiv_brain/dl/data_folds_raw/covar_train_{}.npy'.format(fidx))
    covar_val = np.load('/mnt/Dev_ssd/Dev_ssd/hiv_brain/dl/data_folds_raw/covar_val_{}.npy'.format(fidx))

    print(X_train[0][0])

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    covar_train = covar_train.reshape(covar_train.shape[0], covar_train.shape[1], 1)
    covar_val = covar_val.reshape(covar_val.shape[0], covar_val.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)
    score_train = score_train.reshape(score_train.shape[0], 1)
    score_val = score_val.reshape(score_val.shape[0], 1)

    print(X_train.shape, '<->', np.unique(y_val, return_counts=True))

    covar_train = covar_train.reshape(covar_train.shape[0], covar_train.shape[1])
    covar_val = covar_val.reshape(covar_val.shape[0], covar_val.shape[1])

    print(np.unique(y_train, return_counts=True))
    prediction_model = construct_model(fidx, X_train, covar_train, int(X_train[0].shape[0]*0.5))

    prediction_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse',
        run_eagerly=True,
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ]
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='min')
    mcp_save = ModelCheckpoint(
        '/mnt/Dev_ssd/Dev_ssd/hiv_brain/classification_checkpoints/{}.h5'.format(fidx), save_best_only=True,
        monitor='val_loss', mode='min'
    )
    hist = prediction_model.fit(
        [X_train, covar_train],
        score_train,
        epochs=500,
        batch_size=64,
        shuffle=True,
        validation_data=([X_val, covar_val], score_val),
        validation_batch_size=32,
        callbacks=[ClearMemory(), early_stopping,  mcp_save]
    )
    # plot_history(hist, '/mnt/Dev_ssd/Dev_ssd/hiv_brain/classification_model_checkpoints/{}.jpg'.format(fidx))
    
    accs.append(min(hist.history['val_rmse']))

    del X_train
    del X_val
    del y_train
    del y_val
    del score_train
    del score_val
    del prediction_model
    del hist
    gc.collect()

print(accs)
print(sum(accs) / len(accs), (max(accs) - min(accs) )/ 2, np.std(accs))