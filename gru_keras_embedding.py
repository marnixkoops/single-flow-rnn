import sys
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
from itertools import islice
import collections
import multiprocessing
import functools
from functools import partial
from functools import reduce
import math
import ml_metrics
import h5py
import gc
import logging
import warnings
from datetime import datetime
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.core.protobuf import rewriter_config_pb2

config_proto = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFFrite_options.arithmetic_optimization = off
config_proto.graph_options.rewrite_options.memory_optimization  = off

warnings.simplefilter(action="ignore", category=FutureWarning)

# input
DATA_PATH = "./data/ga_product_sequence_20191013.csv"
INPUT_VAR = "product_sequence"
DATA_CHUNKS = [0]
DATE_FILTER = 20191012  # min visit_date in data is 20191007 and the max is 20191013

N_TOP_PRODUCTS = 500
EMBEDDING_DIM = 32
N_HIDDEN_UNITS = 1500
# DROPOUT = 0.0
LEARNING_RATE = 0.001

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

####################################################################################################
# FUNCTION DEFS
####################################################################################################


def read_raw_data_pandas(file_path, chunk_ids, date_num_from):
    list_with_dt_chuncks = []
    for i in chunk_ids:
        file_name = file_path
        list_with_dt_chuncks.append(pd.read_csv(file_name, header=0, low_memory=False))

    output_raw = pd.concat(list_with_dt_chuncks)
    print("Included columns:", str(output_raw.columns.values))
    print("Raw data\nRows and columns:", str(output_raw.shape))

    output = output_raw[output_raw["visit_date"] > date_num_from][:]
    print("Filtered data\nRows and columns:", str(output.shape))

    return output

####################################################################################################
# PREPARE DATA
####################################################################################################


sequence_df = read_raw_data_pandas(
    file_path=DATA_PATH, chunk_ids=DATA_CHUNKS, date_num_from=DATE_FILTER
)

# take small subset of data for debugging purposes
sequence_df = sequence_df.tail(250000).copy()

sequence_df.head()


tokenizer = keras.preprocessing.text.Tokenizer(num_words=N_TOP_PRODUCTS)
tokenizer.fit_on_texts(sequence_df["product_sequence"])
sequences = tokenizer.texts_to_sequences(sequence_df["product_sequence"])
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='pre')

padded_sequences[:10]


def generate_sliding_window(array, window_size=5, step_size=1):
    shape = (array.size - window_size + 1, window_size)
    strides = array.strides * 2
    window = np.lib.stride_tricks.as_strided(array, strides=strides, shape=shape)[0::step_size]
    return window.copy()


# generate sliding window of sequences with x=4 input products and y=1 target product
window_sequences = np.apply_along_axis(generate_sliding_window, 1, padded_sequences)
window_sequences = np.vstack(window_sequences)  # stack sequences
window_sequences.shape

# delete all sequences with less than 2 products (single x,y pair)
# create a mask for rows that contain more than 5-2=3 elements that are 0
invalid_sequence_mask = (np.sum((window_sequences == 0), axis=1) > 3)
window_sequences = window_sequences[~invalid_sequence_mask].copy()
window_sequences.shape

# split sequences into x=4 input products and y=1 target product
input_sequences = window_sequences[:, :-1]  # matrix of input sequences
target_vector = window_sequences[:, -1]  # vector of targets
input_sequences.shape, target_vector.shape  # check shapes

# split sequences into train-test subsets
test_index = int(TEST_RATIO * len(input_sequences))
X_train, y_train = input_sequences[test_index:], target_vector[test_index:]
X_test, y_test = input_sequences[:test_index], target_vector[:test_index]

X_train.shape, y_train.shape
X_test.shape, y_test.shape

####################################################################################################
# TRAIN MODEL
####################################################################################################


model = keras.Sequential()
model.add(
    keras.layers.Embedding(input_dim=N_TOP_PRODUCTS + 1, output_dim=EMBEDDING_DIM, mask_zero=True)
)
model.add(keras.layers.GRU(N_HIDDEN_UNITS))
model.add(keras.layers.Dense(1, activation="softmax"))
rmsprop_optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
model.compile(loss="binary_crossentropy", optimizer=rmsprop_optimizer, metrics=["accuracy"])
model.summary()

model.fit(input_sequences, target_vector, epochs=3, batch_size=64, validation_split=VAL_RATIO)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

y_pred = model.predict(X_test)
model.predict(X_test).shape[0] == np.sum((y_pred.astype(int)))


# embeddings
my_embeddings = model.get_weights()[0]
my_embeddings.shape

y_test
