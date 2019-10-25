import os
import numpy as np
import pandas as pd
import time
import mlflow
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf
from tensorflow.python.client import device_lib

from ml_metrics import average_precision
from sklearn.metrics import accuracy_score


####################################################################################################
# EXPERIMENT SETTINGS
####################################################################################################

# run mode
DRY_RUN = True  # runs flow on small subset of data for speed and disables mlfow tracking
DOUBLE_DATA = False  # loads two weeks worth of raw data instead of 1 week

# input
DATA_PATH1 = "data/ga_product_sequence_20191013.csv"
DATA_PATH2 = "data/ga_product_sequence_20191020.csv"
INPUT_VAR = "product_sequence"

# constants
# top 6000 products is ~70% of views, 8000 is 80%, 10000 is ~84%, 12000 is ~87%, 15000 is ~90%
N_TOP_PRODUCTS = 1000
embedding_dim = 512
units = 1024
WINDOW_LENGTH = 4  # fixed window size to generare train/validation pairs for training
MIN_PRODUCTS = 3  # sequences with less are considered invalid and removed
DTYPE_GRU = tf.float32

EPOCHS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_STEPS = 5000
DROPOUT = 1
OPTIMIZER = "RMSProp"
CLIP_GRADIENTS = 1.0  # float

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
SHUFFLE_TRAIN_SET = True

# debugging constants
if DRY_RUN:
    N_TOP_PRODUCTS = 200
    EMBED_DIM = 32
    N_HIDDEN_UNITS = 64
    BATCH_SIZE = 64


####################################################################################################
# READ RAW DATA
####################################################################################################

print("[⚡] Running experiment with DRY_RUN: {} and DOUBLE_DATA: {}".format(DRY_RUN, DOUBLE_DATA))

print("\n[⚡] Reading raw input data")

if DRY_RUN:
    sequence_df = pd.read_csv(DATA_PATH1)
    sequence_df = sequence_df.tail(10000).copy()  # take a small subset of data for debugging
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
# PROCESS AND PREPARE DATA
####################################################################################################

t_prep = time.time()  # start timer #1

print("\n[⚡] Tokenizing, padding, filtering & splitting sequences")
print("     Including top {} most popular products".format(N_TOP_PRODUCTS))


def filter_valid_sequences(array, min_items=MIN_PRODUCTS):
    """Short summary.

    Args:
        array (array): Array with sequences encoded as integers.
        min_items (type): Minimum number of products required per sequence.

    Returns:
        type: Array with sequences after filtering valid entries.

    """
    pre_len = len(array)
    valid_sequence_mask = np.sum((array != 0), axis=1) >= min_items  # create mask
    valid_sequences = array[valid_sequence_mask].copy()
    print("     Removed {} invalid sequences".format(pre_len - len(valid_sequences)))
    return valid_sequences


# tokenize, pad and filter sequences
sequences = sequence_df["product_sequence"]


def tokenize(sequences):
    product_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=N_TOP_PRODUCTS)
    product_tokenizer.fit_on_texts(sequences)
    tensor = product_tokenizer.texts_to_sequences(sequences)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="pre")
    return tensor, product_tokenizer


sequences, product_mapping = tokenize(sequences)
sequences = filter_valid_sequences(sequences, min_items=MIN_PRODUCTS)

# split sequences into X, y subsets for training/validation/testing
train_index = int(TRAIN_RATIO * len(sequences))
val_index = train_index + int(VAL_RATIO * len(sequences))
X_train, y_train = sequences[:train_index, :-1], sequences[:train_index, -1]
X_val, y_val = sequences[train_index:val_index, :-1], sequences[train_index:val_index, -1]
X_test, y_test = sequences[val_index:, :-1], sequences[val_index:, -1]
max_length_inp, max_length_targ = sequences.shape[1], 1  # for now we predict output sequence of n=1


print("[⚡] Created dataset dimensions:")
print("     Training X {}, y {}".format(X_train.shape, y_train.shape))
print("     Validation X {}, y {}".format(X_val.shape, y_val.shape))
print("     Testing X {}, y {}".format(X_test.shape, y_test.shape))
print("     Max input length {}, output length {}".format(max_length_inp, max_length_targ))


print("\n[⚡] Generating training batches")
BUFFER_SIZE = len(X_train)
steps_per_epoch = len(X_train) // BATCH_SIZE
vocab_inp_size = N_TOP_PRODUCTS + 1
vocab_tar_size = N_TOP_PRODUCTS + 1

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)  # incomplete last batch is dropped
example_input_batch, example_target_batch = next(iter(dataset))

print("     Batch size: {}".format(BATCH_SIZE))
print("     Steps per epoch X: {}".format(steps_per_epoch))
print(
    "     Input batch shape {}, target batch shape {}".format(
        example_input_batch.shape, example_target_batch.shape
    )
)

print("[⚡] Elapsed time for preparing data: {:.3} seconds".format(time.time() - t_prep))

# clean up memory
del sequence_df, sequences


####################################################################################################
# DEFINE NEURAL SEQUENCE NETWORK ARCHITECTURE
####################################################################################################

print("[⚡] Defining Neural Sequence Network with Bahdanau Attention Mechanism")


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


####################################################################################################
# DEFINE OPTIMIZATION, LOSS, TRAINING & EVALUATION
####################################################################################################

# optimizer
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


# loss function
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index["<start>"]] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    # sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index["<start>"]], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + " "

        if targ_lang.index_word[predicted_id] == "<end>":
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap="viridis")

    fontdict = {"fontsize": 14}

    ax.set_xticklabels([""] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([""] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(result))

    attention_plot = attention_plot[: len(result.split(" ")), : len(sentence.split(" "))]
    plot_attention(attention_plot, sentence.split(" "), result.split(" "))


####################################################################################################
# CREATE NEURAL SEQUENCE NETWORK AND PRINT SUMMARY
####################################################################################################

# encoder
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print("     Encoder output shape: (batch size, seq length, units) {}".format(sample_output.shape))
print("     Encoder Hidden state shape: (batch size, units) {}".format(sample_hidden.shape))

# attention layer
attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, seq length, 1) {}".format(attention_weights.shape))

# decoder
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)), sample_hidden, sample_output)
print("Decoder output shape: (batch_size, vocab size) {}".format(sample_decoder_output.shape))


####################################################################################################
# TRAIN & EVALUATE
####################################################################################################

print("\n[⚡] Starting model training & evaluation")

if not DRY_RUN:
    mlflow.start_run()  # start mlflow run for experiment tracking
    checkpoint_dir = "./checkpoints"  # training checkpoints (Object-based saving)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

t_train = time.time()  # start timer #2

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / steps_per_epoch))
    print("Time taken for 1 epoch {} sec\n".format(time.time() - start))


train_time = time.time() - t_train
print(
    "[⚡] Elapsed time for training on {} sequences: {:.3} minutes".format(
        len(y_train), train_time / 60
    )
)


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
    # mlflow.log_metric("novelty", novelty)
    mlflow.log_metric("Train mins", np.round(train_time / 60), 2)
    mlflow.log_metric("Pred secs", np.round(pred_time))

    # Log executed code
    mlflow.log_artifact("marnix-single-flow-rnn/gru_tf2_attention.py")

    print("[⚡] Elapsed total time: {:.3} minutes".format((time.time() - t_prep) / 60))

    mlflow.end_run()
