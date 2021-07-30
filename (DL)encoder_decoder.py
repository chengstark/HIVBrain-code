import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel("ERROR")
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, MaxPool1D, UpSampling1D, Input, BatchNormalization
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
import time
from tqdm import tqdm



def normalize(array):
    max_of_rows = array.max(axis=1)
    normalized_array = array / max_of_rows[:, np.newaxis]
    return normalized_array


def zero_padding(array):
    zeros_bfr = np.zeros((array.shape[0], 1))
    zeros_aft = np.zeros((array.shape[0], 1))
    padded = np.concatenate((zeros_bfr, array), axis=1)
    padded = np.concatenate((padded, zeros_aft), axis=1)
    return padded


def construct_ed():
    ipt = Input(shape=(25, 1))

    # encoder
    encoded1 = Conv1D(16, 3, activation='relu', padding='same', name='e1')(ipt)
    encoded1 = MaxPool1D(5, padding='same', name='e1-mp1')(encoded1)

    encoded2 = Conv1D(16, 3, activation='relu', padding='same', name='e2')(encoded1)
    encoded2 = MaxPool1D(5, padding='same', name='e2-mp2')(encoded2)

    # bottleneck
    x = Conv1D(32, kernel_size=3, padding='same', activation='linear')(encoded2)
    x = BatchNormalization()(x)

    # decoder
    decoded1 = Conv1D(16, 3, activation='relu', padding='same')(x)
    decoded1 = UpSampling1D(5)(decoded1)

    decoded2 = Conv1D(16, 3, activation='relu', padding='same')(decoded1)
    decoded2 = UpSampling1D(5)(decoded2)

    # output
    decoded = Conv1D(1, 1, activation="sigmoid", padding="same", name="opt")(decoded2)

    model = Model(ipt, decoded)
    print(model.summary())
    return model


def evaluate(npy_name):
    ed = construct_ed()
    ed.load_weights('ed_models/ed.h5')
    ed.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy')
    test_data = np.load('brain-flow-data/2021_0521-BRAIN_Batch_1_npys/{}_PBMC.npy'.format(npy_name))
    pad_test_data = zero_padding(test_data)
    norm_test_data = normalize(pad_test_data)
    result = ed.evaluate(norm_test_data, norm_test_data)
    print('Eval: {} {}'.format(npy_name, result))


if __name__ == '__main__':


    npy_path = '/home/chengstark/Dev/hiv_brain/drug_use/viable_npy/'
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for fidx, splits in enumerate(kf.split(os.listdir(npy_path))):
        train_idx, val_idx = splits
        train_f = [os.listdir(npy_path)[i] for i in train_idx]
        val_f = [os.listdir(npy_path)[i] for i in val_idx]
        
        

        mcp_save = ModelCheckpoint(
            'ed_models/ed.h5', save_best_only=True,
            monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')

        ed = construct_ed()


        x_train = np.load(npy_path+train_f[0])
        for f in train_f[1:]:
            x_train_ = np.load(npy_path+f)
            x_train = np.concatenate((x_train, x_train_),axis=0)

        mean_ = np.mean(x_train, axis=0)
        std_ = np.std(x_train, axis=0)
        x_train = (x_train - mean_) / std_


        x_val = np.load(npy_path+val_f[0])
        for f in val_f[1:]:
            x_val_ = np.load(npy_path+f)
            x_val = np.concatenate((x_val, x_val_),axis=0)

        x_val = (x_val - mean_) / std_

        print(x_train.shape, x_val.shape)

        ed.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
        hist = ed.fit(
            x_train,
            x_train,
            epochs=100,
            batch_size=1024,
            shuffle=True,
            validation_data=(x_val, x_val),
            callbacks=[mcp_save, early_stopping]
        )










