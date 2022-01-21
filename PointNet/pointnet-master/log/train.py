import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

BATCH_SIZE = 128
NUM_POINT = 1024
MAX_EPOCH = 50
BASE_LEARNING_RATE = 0.001
GPU_INDEX = 0
MOMENTUM = 0.9
DECAY_STEP = 100000
DECAY_RATE = 0.5

MODEL = importlib.import_module("pointnet_cls")  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', "pointnet_cls" + '.py')
LOG_DIR = "log"
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('copy %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('copy train.py %s' % LOG_DIR)  # bkp of train procedure
CKPT_DIR = os.path.join(LOG_DIR, "ckpts")
if not os.path.exists(CKPT_DIR): os.mkdir(CKPT_DIR)

MAX_NUM_POINT = 4096
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()


def get_learning_rate_schedule():
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        BASE_LEARNING_RATE,  # Initial learning rate
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    return learning_rate


def random_split(samples, atFraction):
    """
    Perform the train/test split.
    """
    print("atFraction = ", atFraction)
    mask = np.random.rand(len(samples)) < atFraction
    return samples[mask], samples[~mask]


def train():
    model = MODEL.get_model((None, 3), NUM_CLASSES)

    learning_rate = get_learning_rate_schedule()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # load data from h5 files
    def load_h5(h5_filename):
        f = h5py.File(h5_filename, 'r')
        data = f['data'][:]
        label = f['label'][:]
        return data, label

    # load train points and labels
    path = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(path, "data\\modelnet40_ply_hdf5_2048\\ply_data_train_50-50.h5")
    train_points = None
    train_labels = None
    cur_points, cur_labels = load_h5(train_path)
    #cur_points = cur_points.reshape(1, -1, 3)
    #cur_labels = cur_labels.reshape(1, -1)
    if train_labels is None or train_points is None:
        train_labels = cur_labels
        train_points = cur_points
    else:
        train_labels = np.column_stack((train_labels, cur_labels))
        train_points = np.column_stack((train_points, cur_points))

    # load test points and labels
    test_path = os.path.join(path, "data\\modelnet40_ply_hdf5_2048\\ply_data_test_50-50.h5")
    test_points = None
    test_labels = None
    cur_points, cur_labels = load_h5(test_path)
    if test_labels is None or test_points is None:
        test_labels = cur_labels
        test_points = cur_points
    else:
        test_labels = np.column_stack((test_labels, cur_labels))
        test_points = np.column_stack((test_points, cur_points))


    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CKPT_DIR, save_weights_only=False, save_best_only=True),
        tf.keras.callbacks.TensorBoard(LOG_DIR)
    ]

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    model.fit(train_points, train_labels,
              validation_data=(test_points, test_labels),
              steps_per_epoch=len(train_points),
              validation_steps=len(test_points),
              epochs=MAX_EPOCH,
              callbacks=callbacks,
              use_multiprocessing=False)
    tf.saved_model.save(model, "trained_model.pb")



if __name__ == "__main__":
    train()
