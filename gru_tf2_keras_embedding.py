import numpy as np
import pandas as pd
import time

import mlflow
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from ml_metrics import average_precision
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

sns.set_style("darkgrid")
sns.set_context("notebook")
warnings.simplefilter(action="ignore", category=FutureWarning)

# tf.enable_eager_execution()
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

####################################################################################################
# EXPERIMENT SETTINGS
####################################################################################################

# run mode
DRY_RUN = False  # runs flow on small subset of data for speed and disables mlfow tracking
DOUBLE_DATA = False  # loads two weeks worth of raw data instead of 1 week

# input
DATA_PATH1 = "./data/ga_product_sequence_20191013.csv"
DATA_PATH2 = "./data/ga_product_sequence_20191020.csv"
INPUT_VAR = "product_sequence"

# constants
# top 6000 products is ~70% of views, 8000 is 80%, 10000 is ~84%, 12000 is ~87%, 15000 is ~90%
N_TOP_PRODUCTS = 6000
EMBED_DIM = 1024
N_HIDDEN_UNITS = 1024
MIN_PRODUCTS = 3  # sequences with less are considered invalid and removed
WINDOW_LEN = 4  # fixed window size to generare train/validation pairs for training
PRED_LOOKBACK = 4  # number of most recent products used per sequence in the test set to predict on
DTYPE_GRU = tf.float32

N_EPOCHS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 128

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
SHUFFLE_TRAIN_SET = True

# debugging constants
if DRY_RUN:
    N_TOP_PRODUCTS = 250
    N_EPOCHS = 6
    LEARNING_RATE = 0.001
    EMBED_DIM = 32
    N_HIDDEN_UNITS = 128
    BATCH_SIZE = 32

####################################################################################################
# READ RAW DATA
####################################################################################################

print("[⚡] Running experiment with DRY_RUN: {} and DOUBLE_DATA: {}".format(DRY_RUN, DOUBLE_DATA))

print("\n[⚡] Reading raw input data")

if DRY_RUN:
    sequence_df = pd.read_csv(DATA_PATH1)
    sequence_df = sequence_df.tail(int(100000)).copy()  # take a small subset of data for debugging
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
def generate_train_test_pairs(array, input_length=WINDOW_LEN, step_size=1):
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
val_index = int(VAL_RATIO * len(padded_sequences_train))
X_train, y_train = padded_sequences_train[:-val_index, :-1], padded_sequences_train[:-val_index, -1]
X_val, y_val = padded_sequences_train[:val_index, :-1], padded_sequences_train[:val_index, -1]
# how many products should we use per sequence to look back for predicting?
X_test, y_test = padded_sequences_test[:, -(PRED_LOOKBACK + 1) : -1], padded_sequences_test[:, -1]

# # only train-test split (no validation)
# X_train, y_train = padded_sequences_train[:, :-1], padded_sequences_train[:, -1]
# X_test, y_test = padded_sequences_test[:, -5:-1], padded_sequences_test[:, -1]

# drop some rows to fit into batch size
train_index = len(X_train) - len(X_train) % BATCH_SIZE
val_index = len(X_val) - len(X_val) % BATCH_SIZE
test_index = len(X_test) - len(X_test) % BATCH_SIZE
X_train, y_train = X_train[:train_index, :], y_train[:train_index]
X_val, y_val = X_val[:val_index:, :], y_val[:val_index]
X_test, y_test = X_test[:test_index, :], y_test[:test_index]

# targets need to be categorical (OHE) for normal categorical cross entropy loss function in keras
# y_train_cat = keras.utils.to_categorical(y_train, num_classes=None, dtype="int32")
# y_val_cat = keras.utils.to_categorical(y_val, num_classes=None, dtype="int32")
# y_test_cat = keras.utils.to_categorical(y_test, num_classes=None, dtype="int32")

print("[⚡] Generated dataset dimensions:")
print("     Training X {}, y {}".format(X_train.shape, y_train.shape))
print("     Validation X {}, y {}".format(X_val.shape, y_val.shape))
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

if not DRY_RUN:
    mlflow.start_run()  # start mlflow run for experiment tracking
t_train = time.time()  # start timer #2

print("\n[⚡] Training model")
print("     Training for {} Epochs with batch size {}".format(N_EPOCHS, BATCH_SIZE))


def embedding_rnn_model(
    vocab_size=N_TOP_PRODUCTS, embed_dim=EMBED_DIM, num_units=N_HIDDEN_UNITS, batch_size=BATCH_SIZE
):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(  # we can also try hashing instead of embedding
                N_TOP_PRODUCTS, EMBED_DIM, batch_input_shape=[BATCH_SIZE, None], mask_zero=True
            ),
            tf.keras.layers.GRU(
                N_HIDDEN_UNITS,
                return_sequences=False,
                stateful=True,
                recurrent_initializer="glorot_uniform",
            ),
            tf.keras.layers.Dense(N_TOP_PRODUCTS, activation="sigmoid"),
        ]
    )
    return model


model = embedding_rnn_model(
    vocab_size=N_TOP_PRODUCTS, embed_dim=EMBED_DIM, num_units=N_HIDDEN_UNITS, batch_size=BATCH_SIZE
)

# note that OHE targets need categorical_crossentropy
# since we want encoded targets directly we can use sparse_categorical_crossentropy!
model.compile(loss="sparse_categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])

# model description
# model.get_config() # detailed parameter settings
model.summary()

model_history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=N_EPOCHS
)


train_time = time.time() - t_train
print(
    "[⚡] Elapsed time for training on {} sequences: {:.3} minutes".format(
        len(y_train), train_time / 60
    )
)

####################################################################################################
# EVALUATION
####################################################################################################


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


def extract_overlap_per_sequence(X_test, y_pred):
    overlap_items = [
        set(X_test[row]) & set(predicted_sequences_5[row]) for row in range(len(X_test))
    ]
    return overlap_items


def compute_average_novelty(X_test, y_pred):
    overlap_items = extract_overlap_per_sequence(X_test, y_pred)
    overlap_sum = np.sum([len(overlap_items[row]) for row in range(len(overlap_items))])
    average_novelty = 1 - (overlap_sum / (len(X_test) * X_test.shape[1]))
    return average_novelty


t_pred = time.time()
print("\n[⚡] Creating recommendations on test set (logits for all N products / sequence)")
y_pred_probs = model.predict(X_test)
# test_scores = model.evaluate(X_test, y_test, verbose=0)

pred_time = time.time() - t_pred
print("     Elapsed time for predicting {} sequences: {:.3} seconds".format(len(y_test), pred_time))

print("[⚡] Computing metrics on test set containing {} sequences".format(len(y_test)))

# process recomendations, extract top 10 recommendations based on the probabilities
predicted_sequences_10 = np.apply_along_axis(generate_predicted_sequences, 1, y_pred_probs)
predicted_sequences_5 = predicted_sequences_10[:, :5]  # top 5 recommendations
predicted_sequences_3 = predicted_sequences_10[:, :3]  # top 5 recommendations

loss = np.mean(
    keras.losses.sparse_categorical_crossentropy(y_test, y_pred_probs, from_logits=True).numpy()
)
y_pred = np.vstack(predicted_sequences_10[:, 0])  # top 1 recommendation
del y_pred_probs

accuracy = np.round(accuracy_score(y_test, y_pred), 4)
loss = np.round(loss, 4)
map3 = np.round(average_precision.mapk(np.vstack(y_test), predicted_sequences_3, k=3), 4)
map5 = np.round(average_precision.mapk(np.vstack(y_test), predicted_sequences_5, k=5), 4)
map10 = np.round(average_precision.mapk(np.vstack(y_test), predicted_sequences_10, k=10), 4)
coverage = np.round(len(np.unique(y_pred)) / len(np.unique(y_test)), 4)
novelty = np.round(compute_average_novelty(X_test, predicted_sequences_5), 4)

print("     Cross Entropy Loss: {:.4}%".format(loss * 100))
print("     Accuracy @ 1: {:.4}%".format(accuracy * 100))
print("     MAP @ 5: {:.4}%".format(map5 * 100))
print("     MAP @ 3: {:.4}%".format(map3 * 100))
print("     MAP @ 10: {:.4}%".format(map10 * 100))
print("     Coverage: {:.4}%".format(coverage * 100))
print("     Novelty: {:.4}% (recom products not in last viewed items)".format(novelty * 100))

# plot training history results
hist_dict = model_history.history
train_loss_values = hist_dict["loss"]
train_acc_values = hist_dict["accuracy"]
val_loss_values = hist_dict["val_loss"]
val_acc_values = hist_dict["val_accuracy"]
epochs = np.arange(1, N_EPOCHS + 1).astype(int)

plt.close()
validation_plots, ax = plt.subplots(2, 1, figsize=(12, 8))
plt.subplot(211)  # plot loss over epochs
plt.plot(
    epochs, train_loss_values, "deepskyblue", linestyle="dashed", marker="o", label="Train Loss"
)
plt.plot(epochs, val_loss_values, "springgreen", marker="o", label="Val Loss")
plt.plot(epochs[-1], loss, "#16a085", marker="8", markersize=12, label="Test Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("{} loss over Epochs".format(model.loss).upper(), size=13, weight="bold")
plt.legend()

plt.subplot(212)  # plot accuracy over epochs
plt.plot(
    epochs, train_acc_values, "deepskyblue", linestyle="dashed", marker="o", label="Train Accuracy"
)
plt.plot(epochs, val_acc_values, "springgreen", marker="o", label="Val Accuracy")
plt.plot(epochs[-1], accuracy, "#16a085", marker="8", markersize=12, label="Test Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuray (k=1) over Epochs".format(model.loss).upper(), size=13, weight="bold")
plt.legend()
plt.tight_layout()

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

    # check which model we are using
    # if "keras.engine.sequential.Sequential" in str(model):
    #     MODEL = "Keras Embedding GRU"
    # else:
    #     MODEL = "TF1 GRU"

    # Set tags
    mlflow.set_tags({"tf": tf.__version__, "machine": MACHINE, "double_data": DOUBLE_DATA})

    # Log parameters
    mlflow.log_param("n_products", N_TOP_PRODUCTS)
    mlflow.log_param("embed_dim", EMBED_DIM)
    mlflow.log_param("n_hidden_units", N_HIDDEN_UNITS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dtype GRU", DTYPE_GRU)
    mlflow.log_param("window", WINDOW_LEN)
    mlflow.log_param("pred_lookback", PRED_LOOKBACK)
    mlflow.log_param("min_products", MIN_PRODUCTS)

    # Log metrics
    mlflow.log_metric("Cross-Entropy Loss", loss)
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("MAP 3", map3)
    mlflow.log_metric("MAP 5", map5)
    mlflow.log_metric("MAP 10", map10)
    mlflow.log_metric("coverage", coverage)
    mlflow.log_metric("novelty", novelty)
    mlflow.log_metric("Train mins", np.round(train_time / 60), 2)
    mlflow.log_metric("Pred secs", np.round(pred_time))

    # Log artifacts
    mlflow.log_artifact("./gru_tf2_keras_embedding.py")  # log executed code
    mlflow.log_artifact("validation plots", validation_plots)  # log validation plots

    print("[⚡] Elapsed total time: {:.3} minutes".format((time.time() - t_prep) / 60))

    mlflow.end_run()
