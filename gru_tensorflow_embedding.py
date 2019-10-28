import numpy as np
import pandas as pd
import time

import mlflow
import warnings

from ml_metrics import average_precision
from sklearn.metrics import accuracy_score

warnings.simplefilter(action="ignore", category=FutureWarning)
with warnings.catch_warnings():  # avoid futurewarnings since we a lot of deprecated stuff
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.python.client import device_lib
    from tensorflow.contrib.learn.python.learn import monitors as monitor_lib

tf.enable_eager_execution()
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


####################################################################################################
# EXPERIMENT SETTINGS
####################################################################################################

# run mode
DRY_RUN = True  # runs flow on small subset of data for speed and disables mlfow tracking
DOUBLE_DATA = True  # loads two weeks worth of raw data instead of 1 week

# input
DATA_PATH1 = "marnix-single-flow-rnn/data/ga_product_sequence_20191013.csv"
DATA_PATH2 = "marnix-single-flow-rnn/data/ga_product_sequence_20191020.csv"
INPUT_VAR = "product_sequence"

# constants
# top 6000 products is ~70% of views, 8000 is 80%, 10000 is ~84%, 12000 is ~87%, 15000 is ~90%
N_TOP_PRODUCTS = 10000
EMBED_DIM = 2048
N_HIDDEN_UNITS = 4096
WINDOW_LENGTH = 4  # fixed window size to generare train/validation pairs for training
MIN_PRODUCTS = 3  # sequences with less are considered invalid and removed
DTYPE_GRU = tf.float32

LEARNING_RATE = 0.001
EPOCHS = 3
BATCH_SIZE = 256
DROPOUT = 1
OPTIMIZER = "RMSProp"
CLIP_GRADIENTS = 1.0  # float

TRAIN_RATIO = 0.85
# VAL_RATIO = 0.1
TEST_RATIO = 0.15
SHUFFLE_TRAIN_SET = True

# debugging constants
if DRY_RUN:
    N_TOP_PRODUCTS = 100
    EMBED_DIM = 32
    N_HIDDEN_UNITS = 64
    BATCH_SIZE = 16


####################################################################################################
# READ RAW DATA
####################################################################################################

print("[⚡] Running experiment with DRY_RUN: {} and DOUBLE_DATA: {}".format(DRY_RUN, DOUBLE_DATA))

print("\n[⚡] Reading raw input data")

if DRY_RUN:
    sequence_df = pd.read_csv(DATA_PATH1)
    sequence_df = sequence_df.tail(2500).copy()  # take a small subset of data for debugging
elif DOUBLE_DATA:
    sequence_df = pd.read_csv(DATA_PATH1)
    sequence_df2 = pd.read_csv(DATA_PATH2)
    sequence_df = sequence_df.append(sequence_df2)  # Create bigger dataset (2 weeks instead of 1)
    del sequence_df2
else:
    sequence_df = pd.read_csv(DATA_PATH1)

MIN_DATE, MAX_DATE = sequence_df["visit_date"].min(), sequence_df["visit_date"].max()

print("     Data contains {} sequences from {} to {}".format(len(sequence_df), MIN_DATE, MAX_DATE))


####################################################################################################
# PREPARE DATA FOR MODELING
####################################################################################################

t_prep = time.time()  # start timer #1

print("\n[⚡] Tokenizing, padding, filtering & splitting sequences")
print("     Including top {} most popular products".format(N_TOP_PRODUCTS))
# define tokenizer to encode sequences while including N most popular items (occurence)
tokenizer = keras.preprocessing.text.Tokenizer(num_words=N_TOP_PRODUCTS)
# encode sequences
tokenizer.fit_on_texts(sequence_df["product_sequence"])
sequences = tokenizer.texts_to_sequences(sequence_df["product_sequence"])


# pre-pad sequences with 0's, length is based on longest present sequence
# this is required to transform the variable length sequences into equal train-test pairs
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="pre")


def filter_valid_sequences(array, min_items=MIN_PRODUCTS):
    """Short summary.

    Args:
        array (array): Array with sequences encoded as integers.
        min_items (type): Minimum number of products required per sequence.

    Returns:
        type: Array with sequences after filtering valid entries.

    """
    pre_len = len(array)
    array = array[array[:, -1] != 0].copy()  # remove all sequences that end in a 0
    valid_sequence_mask = np.sum((array != 0), axis=1) >= min_items  # create mask
    valid_sequences = array[valid_sequence_mask].copy()
    print("     Removed {} invalid sequences".format(pre_len - len(valid_sequences)))
    return valid_sequences


padded_sequences = filter_valid_sequences(padded_sequences, min_items=MIN_PRODUCTS)

# split into train/test subsets before reshaping sequence for training/validation
# since we process some sequences of a single user into multiple train/validation pairs
# while actual recommendations (testing) should be done per user equal as in practice
test_index = int(TEST_RATIO * len(padded_sequences))
padded_sequences_train = padded_sequences[test_index:].copy()
padded_sequences_test = padded_sequences[:test_index].copy()
print("     Training & evaluating model on {} sequences".format(len(padded_sequences_train)))
print("     Testing recommendations on {} sequences".format(len(padded_sequences_test)))

# clean up memory
del sequence_df, sequences, padded_sequences


# reshape sequences for model training/validation
def generate_train_test_pairs(array, input_length=WINDOW_LENGTH, step_size=1):
    """Reshapes an input matrix with equal length padded sequences into a matrix with desired size.
    Output shape is based on the input_length. Note that the output width is input_length + 1 since
    we later take the last column of the matrix to obtain an input matrix of width input_length and
    a vector with corresponding targets (next item in the sequence).
    Args:
        array (array): Input matrix with equal length padded sequences.
        input_length (int): Size of sliding window, equal to desired length of input sequence.
        step_size (int): Can be used to skip.
    Returns:
        array: Reshaped matrix with # columns equal to input_length + 1 (input + target item).
    """
    shape = (array.size - input_length + 2, input_length + 1)
    strides = array.strides * 2
    window = np.lib.stride_tricks.as_strided(array, strides=strides, shape=shape)[0::step_size]
    return window.copy()


print("[⚡] Reshaping into train-test sequences with fixed window size for training")
# generate sliding window of sequences with x=4 input products and y=1 target product
padded_sequences_train = np.apply_along_axis(generate_train_test_pairs, 1, padded_sequences_train)
padded_sequences_train = np.vstack(padded_sequences_train).copy()  # stack sequences
print("     Generated {} sequences for training/validation".format(len(padded_sequences_train)))

# filter sequences, note that due to reshaping again invalid sequences can be generated
padded_sequences_train = filter_valid_sequences(padded_sequences_train, min_items=MIN_PRODUCTS)

# shuffle training sequences randomly (rows, not within sequences ofcourse)
if SHUFFLE_TRAIN_SET:
    np.random.shuffle(padded_sequences_train)  # shuffles in-place

# split sequences into subsets for training/validation/testing
# to predict 1 sequence per user in the test set we predict based on the last 4 items
# val_index = int(VAL_RATIO * len(padded_sequences_train))
# X_train, y_train = padded_sequences_train[:-val_index, :-1], padded_sequences_train[:-val_index, -1]
# X_val, y_val = padded_sequences_train[:val_index, :-1], padded_sequences_train[:val_index, -1]
# X_test, y_test = padded_sequences_test[:, -5:-1], padded_sequences_test[:, -1]

# only train-test split for now (no validation)
X_train, y_train = padded_sequences_train[:, :-1], padded_sequences_train[:, -1]
X_test, y_test = padded_sequences_test[:, -5:-1], padded_sequences_test[:, -1]

print("[⚡] Generated dataset dimensions:")
print("     Training X {}, y {}".format(X_train.shape, y_train.shape))
# print("     Validation X {}, y {}".format(X_val.shape, y_val.shape))
print("     Testing X {}, y {}".format(X_test.shape, y_test.shape))

# clean up memory
del padded_sequences_train, padded_sequences_test

print(
    "[⚡] Elapsed time for tokenizing, padding, filtering & splitting: {:.3} seconds".format(
        time.time() - t_prep
    )
)


####################################################################################################
# DEFINE AND TRAIN RECURRENT NEURAL NETWORK
####################################################################################################
# https://machinelearnings.co/tensorflow-text-classification-615198df9231

print("\n[⚡] Starting model training & evaluation")
steps_per_epoch = len(X_train) // BATCH_SIZE
steps = EPOCHS * steps_per_epoch
print("     Training for {} Epochs ({} steps with batch size {})".format(EPOCHS, steps, BATCH_SIZE))

if not DRY_RUN:
    mlflow.start_run()  # start mlflow run for experiment tracking

t_train = time.time()  # start timer #2

# CONSIDER HASHING INSTEAD OF EMBEDDINGS


def embedding_rnn_model(
    input_sequence,
    target,
    vocab_size=N_TOP_PRODUCTS,
    embed_dim=EMBED_DIM,
    num_units=N_HIDDEN_UNITS,
    dtype=DTYPE_GRU,
    dropout=DROPOUT,
    optimizer=OPTIMIZER,
    clip_gradients=CLIP_GRADIENTS,
):
    """Defines a Recurrent Neural Network with GRU cells for seq2seq modeling purposes.
    Input sequences are embedded to reduce (1) data dimensionality and (2) required network complexity.
    Both (1) and (2) greatly reduce computational effort of model training and prediction.
    Args:
        input_sequence (array): sequence of input items.
        target (array): The next item after the input_sequence.
    Returns:
        type: A Recurrent Neural Network model.
    """
    # Convert indexes of words into embeddings. This creates embeddings matrix of
    # [n_words, EMBEDDING_SIZE] and then maps word indexes of the sequence into
    # [batch_size, sequence_length, EMBEDDING_SIZE].
    embeddings = tf.contrib.layers.embed_sequence(
        input_sequence, vocab_size=N_TOP_PRODUCTS, embed_dim=EMBED_DIM, trainable=True
    )
    # embeddings_list = tf.unstack(embedding, axis=1)  # messes up the rank of the tensor

    # tf.contrib.rnn.GRUBlockCellV2 optimzed for CPU, tf.contrib.cudnn_rnn.CudnnGRU for GPU
    cell = tf.contrib.rnn.GRUCell(N_HIDDEN_UNITS)
    # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=DROPOUT)

    # Create an unrolled Recurrent Neural Networks to length of max_sequence_length and passes
    # word_list as inputs for each unit.
    output, state = tf.nn.dynamic_rnn(
        cell, embeddings, dtype=DTYPE_GRU, time_major=False
    )  # time_major = False means input is of shape [batch_size, sequence_length, EMBEDDING_SIZE]
    # instead of [sequence_length, batch_size, EMBEDDING_SIZE]

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the neural network of
    # last step) and pass it as features to fully connected layer to output probabilities per class.
    # Note that the target is required to be one-hot-encoded to outut logits per class.
    target = tf.one_hot(target, N_TOP_PRODUCTS, 1, 0)
    logits = tf.contrib.layers.fully_connected(state, N_TOP_PRODUCTS, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        clip_gradients=CLIP_GRADIENTS,
    )

    return ({"class": tf.argmax(logits, 1), "prob": tf.nn.softmax(logits)}, loss, train_op)


model = tf.contrib.learn.Estimator(model_fn=embedding_rnn_model)
model.fit(X_train, y_train, max_steps=steps, batch_size=BATCH_SIZE)

train_time = time.time() - t_train
print(
    "[⚡] Elapsed time for training on {} sequences: {:.3} minutes".format(
        len(y_train), train_time / 60
    )
)

t_pred = time.time()

# switching to CPU to avoid memory issues
# tf.device("/cpu:0")

print("\n[⚡] Creating recommendations on test set (probs for all N products per sequence)")
# y_pred_class = np.array([p["class"] for p in model.predict(X_test, as_iterable=True)])
y_pred_probs = np.array(  # memory efficient generator with batches to avoid memory issues
    [
        p["prob"]
        for p in model.predict(X_test, as_iterable=True, batch_size=1024, iterate_batches=True)
    ]
)

pred_time = time.time() - t_pred
print("     Elapsed time for predicting {} sequences: {:.3} seconds".format(len(y_test), pred_time))


print("[⚡] Computing metrics on test set containing {} sequences".format(len(y_test)))


def generate_predicted_sequences(y_pred_probs, output_length=10):
    """Function to extract predicted output sequences. Output is based on the predicted logit values
    where the highest probability corresponds to the first recommended item and so forth.
    Output positions are based on probability from high to low so the output sequence is ordered.
    To be used for obtaining multiple product recommendations and calculating MAP@K values.
    Args:
        y_pred_probs (array): Description of parameter `y_pred_probs`.
        output_length (int): Description of parameter `output_length`.
    Returns:
        array: Description of returned object.
    """
    # obtain indices of highest logit values, the position corresponds to the encoded item
    ind_of_max_logits = np.argpartition(y_pred_probs, -output_length)[-output_length:]
    # order the sequence, sorting the negative values ascending equals sorting descending
    ordered_predicted_sequences = ind_of_max_logits[np.argsort(-y_pred_probs[ind_of_max_logits])]

    return ordered_predicted_sequences.copy()


y_pred_probs = np.vstack(y_pred_probs)  # stack batches to (y_test, N_TOP_PRODUCTS)
# y_pred_probs[:2].shape
# print("     Allocating memory for big prediction matrix")
# # process recomendations
# # allocate memory space for big stacked array to avoid memory issues
# rows = len(y_test)
# y_pred_container = np.empty((rows, N_TOP_PRODUCTS), dtype=np.float32)
# y_pred_container_1 = y_pred_container[: rows // 2]
# y_pred_container_2 = y_pred_container[rows // 2 :]
# y_pred_container_1[:] = np.vstack(
#     y_pred_probs[: rows // 2]
# )  # stack batches to (y_test, N_TOP_PRODUCTS)
# y_pred_container_2[:] = np.vstack(
#     y_pred_probs[rows // 2 :]
# )  # stack batches to (y_test, N_TOP_PRODUCTS)

predicted_sequences_10 = np.apply_along_axis(generate_predicted_sequences, 1, y_pred_probs)  # top10
predicted_sequences_5 = predicted_sequences_10[:, :5]  # top 5
predicted_sequences_3 = predicted_sequences_10[:, :3]  # top 3
y_pred = np.vstack(predicted_sequences_10[:, 0])  # top 1 recommendation (predicted next click)

score = np.round(accuracy_score(y_test, y_pred), 4)
map3 = np.round(average_precision.mapk(np.vstack(y_test), predicted_sequences_3, k=3), 4)
map5 = np.round(average_precision.mapk(np.vstack(y_test), predicted_sequences_5, k=5), 4)
map10 = np.round(average_precision.mapk(np.vstack(y_test), predicted_sequences_10, k=10), 4)
score = np.round(accuracy_score(y_test, y_pred), 4)
# cross_entropy = np.round(log_loss(y_test, predicted_sequences_5[:,0]), 4)
coverage = np.round(len(np.unique(y_pred)) / len(np.unique(y_test)), 4)
# novelty =


print("     Accuracy @ 1: {:.4}%".format(score * 100))
print("     MAP @ 5: {:.4}%".format(map5 * 100))
print("     MAP @ 10: {:.4}%".format(map10 * 100))
# print("     Cross Entropy Loss: {:.4}%".format(cross_entropy))
print("     Coverage @ 1: {:.4}%".format(coverage * 100))
# print("     Novelty @ 5: {:.4}% (recom products not in input)".format(novelty * 100))


####################################################################################################
# EXPERIMENT TRACKING WITH MLFLOW
####################################################################################################

if not DRY_RUN:
    print("[⚡] Logging experiment to mlflow")

    # check if we are runnnig on GPU (Cloud) or CPU (local)
    if "GPU" in str(device_lib.list_local_devices()):
        MACHINE = "cloud"
    else:
        MACHINE = "local"

    # Set tags
    mlflow.set_tags({"machine": MACHINE, "tf": tf.__version__, "double_data": DOUBLE_DATA})

    # Log parameters
    mlflow.log_param("n_products", N_TOP_PRODUCTS)
    mlflow.log_param("embed_dim", EMBED_DIM)
    mlflow.log_param("n_hidden_units", N_HIDDEN_UNITS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("dropout", DROPOUT)
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("clip_gradients", CLIP_GRADIENTS)
    mlflow.log_param("dtype GRU", DTYPE_GRU)
    mlflow.log_param("window", WINDOW_LENGTH)
    mlflow.log_param("min_products", MIN_PRODUCTS)

    # Log metrics
    mlflow.log_metric("Accuracy", score)
    mlflow.log_metric("MAP 3", map3)
    mlflow.log_metric("MAP 5", map5)
    mlflow.log_metric("MAP 10", map10)
    #    mlflow.log_metric("Cross Entropy", loss)
    mlflow.log_metric("coverage", coverage)
    # mlflow.log_metric("novelty", novelty)
    mlflow.log_metric("Train mins", np.round(train_time / 60), 2)
    mlflow.log_metric("Pred secs", np.round(pred_time))

    # Log executed code
    mlflow.log_artifact("marnix-single-flow-rnn/gru_tensorflow_embedding.py")

    print("[⚡] Elapsed total time: {:.3} minutes".format((time.time() - t_prep) / 60))

    mlflow.end_run()
