import numpy as np
import pandas as pd
import time
import mlflow
import warnings

from ml_metrics import average_precision
from sklearn.metrics import accuracy_score, log_loss

warnings.simplefilter(action="ignore", category=FutureWarning)
with warnings.catch_warnings():  # avoid futurewarnings since we a lot of deprecated stuff
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.python.client import device_lib

tf.enable_eager_execution()
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


####################################################################################################
# EXPERIMENT SETTINGS
####################################################################################################

# run mode
DRY_RUN = False  # runs flow on small subset of data for speed and disables mlfow tracking
DOUBLE_DATA = False  # loads two weeks worth of raw data instead of 1 week

# input
DATA_PATH1 = "./data/ga_product_sequence_20191013.csv"
DATA_PATH2 = "./data/ga_product_sequence_20191013.csv"
INPUT_VAR = "product_sequence"

# constants
N_TOP_PRODUCTS = 6000
EMBED_DIM = 512
N_HIDDEN_UNITS = 3000
WINDOW_LENGTH = 4  # fixed window size to generare train/validation pairs for training
MIN_PRODUCTS = 3  # sequences with less are considered invalid and removed
DTYPE_GRU = tf.float32

N_EPOCHS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 256
MAX_STEPS = 5e3
DROPOUT = 1
OPTIMIZER = "RMSProp"
CLIP_GRADIENTS = 1.0  # float

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# debugging constants
if DRY_RUN:
    N_TOP_PRODUCTS = 100
    EMBED_DIM = 32
    N_HIDDEN_UNITS = 64
    BATCH_SIZE = 16
    N_EPOCHS = 1


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

# split sequences into subsets for training/validation/testing
# to predict 1 sequence per user in the test set we predict based on the last 4 items
val_index = int(VAL_RATIO * len(padded_sequences_train))
X_train, y_train = padded_sequences_train[:-val_index, :-1], padded_sequences_train[:-val_index, -1]
X_val, y_val = padded_sequences_train[:val_index, :-1], padded_sequences_train[:val_index, -1]
X_test, y_test = padded_sequences_test[:, -5:-1], padded_sequences_test[:, -1]

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

print("\n[⚡] Starting model training & evaluation")

if not DRY_RUN:
    mlflow.start_run()  # start mlflow run for experiment tracking
t_train = time.time()  # start timer #2


def embedding_rnn_model(
    vocab_size=N_TOP_PRODUCTS, embed_dim=EMBED_DIM, num_units=N_HIDDEN_UNITS, batch_size=BATCH_SIZE
):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                N_TOP_PRODUCTS, EMBED_DIM, batch_input_shape=[BATCH_SIZE, None]
            ),
            tf.keras.layers.GRUCell(
                N_HIDDEN_UNITS,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
            ),
            tf.keras.layers.Dense(N_TOP_PRODUCTS),
        ]
    )
    return model


model = embedding_rnn_model(
    vocab_size=N_TOP_PRODUCTS,
    embedding_dim=EMBED_DIM,
    rnn_units=N_HIDDEN_UNITS,
    batch_size=BATCH_SIZE,
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="RMSprop",
    metrics=["accuracy", "categorical_crossentropy"],
)

model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=N_EPOCHS)

train_time = time.time() - t_train
print(
    "[⚡] Elapsed time for training on {} sequences: {:.3} minutes".format(
        len(y_train), train_time / 60
    )
)

t_pred = time.time()
print("\n[⚡] Creating recommendations on test set (logits for all N products / sequence)")
# y_pred_class = np.array([p["class"] for p in model.predict(X_test, as_iterable=True)])
y_pred_probs = np.array([p["prob"] for p in model.predict(X_test, as_iterable=True)])
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


# process recomendations
predicted_sequences_10 = np.apply_along_axis(generate_predicted_sequences, 1, y_pred_probs)
predicted_sequences_5 = predicted_sequences_10[:, :5]  # top 5 recommendations
predicted_sequences_3 = predicted_sequences_10[:, :3]  # top 5 recommendations
y_pred = np.vstack(predicted_sequences_10[:, 0])  # top 1 recommendation

score = np.round(accuracy_score(y_test, y_pred), 4)
map3 = np.round(average_precision.mapk(np.vstack(y_test), predicted_sequences_3, k=3), 4)
map5 = np.round(average_precision.mapk(np.vstack(y_test), predicted_sequences_5, k=5), 4)
map10 = np.round(average_precision.mapk(np.vstack(y_test), predicted_sequences_10, k=10), 4)
score = np.round(accuracy_score(y_test, y_pred), 4)
# cross_entropy = np.round(log_loss(y_test, predicted_sequences_5[:,0]), 4)
coverage = np.round(len(np.unique(y_pred)) / len(np.unique(y_test)), 4)
# recom_novelty = sum([(predicted_sequences[i] not in X_test[i]) for i in range(len(y_test))]) / len(y_test)

print("     Accuracy @ 1: {:.4}%".format(score * 100))
print("     MAP @ 5: {:.4}%".format(map5 * 100))
print("     MAP @ 10: {:.4}%".format(map10 * 100))
# print("     Cross Entropy Loss: {:.4}%".format(cross_entropy))
print("     Coverage @ 1: {:.4}%".format(coverage * 100))
# print("     Novelty @ 1: {:.4}% (recom products not in input)".format(recom_novelty * 100))


####################################################################################################
# EXPERIMENT TRACKING WITH MLFLOW
####################################################################################################

if not DRY_RUN:
    print("[⚡] Logging experiment to mlflow")

    # Set tags
    mlflow.set_tags({"double_data": DOUBLE_DATA, "tf": tf.__version__})

    # Log parameters
    mlflow.log_param("n_products", N_TOP_PRODUCTS)
    mlflow.log_param("embed_dim", EMBED_DIM)
    mlflow.log_param("n_hidden_units", N_HIDDEN_UNITS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("max_steps", MAX_STEPS)
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
    #    mlflow.log_metric("novelty", novelty)
    mlflow.log_metric("Train mins", np.round(train_time / 60), 2)
    mlflow.log_metric("Pred secs", np.round(pred_time))

    # Log executed code
    mlflow.log_artifact("gru_tf2_keras_embedding.py")

    print("[⚡] Elapsed total time: {:.3} minutes".format((time.time() - t_prep) / 60))

    mlflow.end_run()
