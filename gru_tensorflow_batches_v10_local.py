####################################################################################################
# SETTINGS
####################################################################################################

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
from tensorflow.python.client import device_lib
from sklearn.metrics import accuracy_score


warnings.simplefilter(action="ignore", category=FutureWarning)

# input
DATA_PATH = "./data/ga_product_sequence_20191013.csv"
INPUT_VAR = "product_sequence"
DATA_CHUNKS = [0]
DATE_FILTER = 20191006  # min visit_date in data is 20191007 and the max is 20191013

# encoding
TOP_POPULAR_PRODUCTS = 6000
REMEMBER_WINDOW = 4
MOVING_WINDOW = 4

# model
HIDDEN_SIZE = 1
NUM_UNITS = 9000
BATCH_SIZE = 128

# training
NUM_EPOCHS = 6
eval_pct = 0.1
test_pct = 0.2
tmp_value = 960000  # large int used to define number of batches
batches_evaluation = 6
batches_eval_size = 8000
prediction_batch = 1000


####################################################################################################
# DEBUGGING MODE
####################################################################################################

DRY_RUN = True

if DRY_RUN:
    TOP_POPULAR_PRODUCTS = 1000
    NUM_EPOCHS = 2
    NUM_UNITS = 9000
    tmp_value = 960000
    batches_evaluation = 6
    batches_eval_size = 8000
    prediction_batch = 100


####################################################################################################
# DATA
####################################################################################################
# query used to get data, note that the length > 6 prefilters sequences with less than 1 product
# as the sequence is a string and each product_id has 6 characters
DATA_QUERY = """SELECT
  coolblue_cookie_id,
  visit_date,
  product_sequence
  FROM
  `coolblue-bi-platform-prod.recommender_systems.ga_product_sequence`
  WHERE
  DATE(_PARTITIONTIME) = "2019-10-13"
  AND LENGTH(product_sequence) > 6"""


####################################################################################################
# FUNCTION DEFS
####################################################################################################


class Printer:
    """
    Print things to stdout on one line dynamically
    """

    def __init__(self, data):

        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


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


def get_available_processing_units():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def extract_sequences_from_column(column):
    return list(map(lambda x: x.split(","), column.tolist()))


def extract_top_n(list_of_list_products, n_popular):

    text_qa_raw = list(itertools.chain.from_iterable(list_of_list_products))
    text_qa = [e for e in text_qa_raw if e not in ("")]

    products_count = collections.Counter(text_qa).most_common(n_popular)
    products_in_scope = [x[0] for x in products_count]

    print("Top {} products extracted".format(str(len(products_in_scope))))
    return products_in_scope


def clean_list(list_target, elements_in_scope):
    return [prod for prod in list_target if prod in elements_in_scope]


def clean_list_par(list_target, products_in_scope):

    pool = multiprocessing.Pool()
    multipros_pars_added = partial(clean_list, elements_in_scope=products_in_scope)
    clean_list_of_products = pool.map(multipros_pars_added, list_target)
    pool.close()
    pool.join()

    return clean_list_of_products


def extract_top_n_products(prob_array, npos):
    products_idx = prob_array.argsort()[-npos:][::-1]
    output = []
    for i in range(npos):
        output.append(reverse_dictionary.get(products_idx[i]))
    # output = reverse_dictionary.get(pointing_value[0][0])
    return output


def extract_product(bol_array):
    pro_pos = int(np.where(bol_array)[0])
    return reverse_dictionary.get(pro_pos)


def mean_avg_precision_k(tupla, k=5):
    return ml_metrics.apk(tupla[0], tupla[1], k)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)

    def f1():
        return 0.5 * tf.square(residual)

    def f2():
        return delta * residual - 0.5 * tf.square(delta)

    return tf.cond(residual < delta, f1, f2)


def products_enconding(list_with_products, dictionar):
    return [dictionar.get(p) for p in list_with_products]


def products_deconding(list_with_products, reverse_dictionary):
    return [reverse_dictionary.get(p) for p in list_with_products]


def encode_sequences(list_products):

    products_totals = list(itertools.chain.from_iterable(list_products))

    count = collections.Counter(products_totals)
    dictionary = dict()
    for prod in count:
        dictionary[prod] = len(dictionary)

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    pool = multiprocessing.Pool()
    multipros_pars_added = partial(products_enconding, dictionar=dictionary)
    data = pool.map(multipros_pars_added, list_products)
    pool.close()
    pool.join()

    return data, dictionary, reversed_dictionary


def encode_sequences_pred(list_products, dictionar):

    pool = multiprocessing.Pool()
    multipros_pars_added = partial(products_enconding, dictionar=dictionary)
    data = pool.map(multipros_pars_added, list_products)
    pool.close()
    pool.join()

    return data


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def extract_boolean_position_x_y_comp_h5py(
    list_with_products, remem_window, mov_window, vocabulary_size
):
    y_elements_chunk = []
    x_elements_chunk = []
    for i in range(len(list_with_products) - 1):
        window_size = min(i + 1, remem_window)
        target_elem = list_with_products[i + 1]

        x_element_chunk_partial = []
        np_bool_y = np.zeros(vocabulary_size, dtype=bool)

        np_bool_y.flat[target_elem] = True

        prod_in_scope = i + 1
        past_elements = list(window(list_with_products[:prod_in_scope], window_size))[-mov_window:]

        x_element_chunk_partial = []
        for j in range(mov_window):
            np_bool_x = np.zeros(vocabulary_size, dtype=bool)
            if len(past_elements) - j > 0:
                np_bool_x.flat[list(past_elements[j])] = True
            x_element_chunk_partial.append(np_bool_x)
        x_boolean_partial = np.vstack(reversed(x_element_chunk_partial))

        x_elements_chunk.append(x_boolean_partial)
        y_elements_chunk.append(np_bool_y)

    y_boolean = np.vstack(y_elements_chunk)
    x_boolean = np.stack(x_elements_chunk)

    return x_boolean, y_boolean


def extract_boolean_position_x_y_comp_pred_h5py(
    list_with_products, remem_window, mov_window, vocabulary_size
):
    x_elements_chunk = []
    for i in range(len(list_with_products)):

        window_size = min(i + 1, remem_window)

        x_element_chunk_partial = []

        prod_in_scope = i + 1
        past_elements = list(window(list_with_products[:prod_in_scope], window_size))[-mov_window:]

        for j in range(mov_window):
            np_bool_x = np.zeros(vocabulary_size, dtype=bool)
            if len(past_elements) - j > 0:
                np_bool_x.flat[list(past_elements[j])] = True
            x_element_chunk_partial.append(np_bool_x)
        x_boolean_partial = np.vstack(reversed(x_element_chunk_partial))

        x_elements_chunk.append(x_boolean_partial)

    x_boolean = np.stack(x_elements_chunk)[-1:]  # just the last one

    return x_boolean


def construct_x_y_h5py(data_to_tabulate, remem_window, moving_window, vocabulary_size):
    pool = multiprocessing.Pool(6)
    multipros_pars_added = partial(
        extract_boolean_position_x_y_comp_h5py,
        remem_window=remem_window,
        mov_window=moving_window,
        vocabulary_size=vocabulary_size,
    )
    dat_tabulated = pool.map(multipros_pars_added, data_to_tabulate)
    pool.close()
    pool.join()

    dat_X = [x[0] for x in dat_tabulated]
    dat_Y = [x[1] for x in dat_tabulated]

    del dat_tabulated

    dat_X_array = np.vstack(dat_X)
    dat_Y_array = np.vstack(dat_Y)

    return dat_X_array, dat_Y_array


def construct_x_y_h5py_pred(data_to_tabulate, remem_window, moving_window, vocabulary_size):
    pool = multiprocessing.Pool(6)
    multipros_pars_added = partial(
        extract_boolean_position_x_y_comp_pred_h5py,
        remem_window=remem_window,
        mov_window=moving_window,
        vocabulary_size=vocabulary_size,
    )
    dat_tabulated = pool.map(multipros_pars_added, data_to_tabulate)
    pool.close()
    pool.join()

    dat_X_array = np.vstack(dat_tabulated)

    return dat_X_array


def join_integers(list_of_ints):
    return ",".join(str(x) for x in list_of_ints)


####################################################################################################
# TRAIN / TEST FLOW
####################################################################################################

# create a logfile for today, also add logging to stdout
logging.handlers = []  # close loggers
logging.basicConfig(
    filename="./logs/single_flow_rnn_{}.log".format(datetime.now().date()), level=logging.INFO
)
stderr_logger = logging.StreamHandler()
stderr_logger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(stderr_logger)
logging.info("{}: Created log file".format(datetime.now()))
logging.info("TensorFlow version: {}".format(tf.__version__))

# run mode alert
if DRY_RUN:
    logging.info(
        "{}: DRY_RUN is True: running on a small subset of data for debugging purposes".format(
            datetime.now()
        )
    )
else:
    logging.info(
        "{}: DRY_RUN is False: running on full dataset: {}".format(datetime.now(), DATA_PATH)
    )


# we load the data and filter it by date
df_filtered = read_raw_data_pandas(
    file_path=DATA_PATH, chunk_ids=DATA_CHUNKS, date_num_from=DATE_FILTER
)

# take small subset of data for debugging purposes
if DRY_RUN:
    df_filtered = df_filtered.tail(1000000).copy()

# these are all the sequences in the filtered dataset
all_questions = extract_sequences_from_column(df_filtered[INPUT_VAR])

# we focus only in the top-N products
popular_products_scope = extract_top_n(
    list_of_list_products=all_questions, n_popular=TOP_POPULAR_PRODUCTS
)


# the sequences are cleaned of those products out of the top list
clean_list_of_products = clean_list_par(
    list_target=all_questions, products_in_scope=popular_products_scope
)

# check how many products where dropped from the data (see coverage of TOP_POPULAR_PRODUCTS)
# result is roughly 70%
top_n_product_coverage = (
    np.concatenate(clean_list_of_products).ravel().shape[0]
    / np.concatenate(all_questions).ravel().shape[0]
)

logging.info(
    "Click coverage of top {} most popular products: {:.0}".format(
        TOP_POPULAR_PRODUCTS, top_n_product_coverage
    )
)

# valid sequences are those that at least have 2 products in its sequence that belong to the top N popular products
sequences_valid = [s for s in clean_list_of_products if len(s) > 1]
del all_questions, clean_list_of_products  # releasing memory
gc.collect()

# this is the data encoded prepared for extracting the features in batches
data_encoded, dictionary, reverse_dictionary = encode_sequences(sequences_valid)

test_sample_size = round(test_pct * len(data_encoded))
eval_sample_size = round(eval_pct * (len(data_encoded) - test_sample_size))
np.random.seed(seed=8868)
idx_evaluation = np.random.choice(
    (len(data_encoded) - test_sample_size), eval_sample_size, replace=False
)
idx_train_eval_set = list(set(range((len(data_encoded) - test_sample_size))) - set(idx_evaluation))

data_encoded_train_eval = data_encoded[:-test_sample_size]

# these are the three sets of sequences
data_encoded_test = data_encoded[-test_sample_size:]
data_encoded_eval = [data_encoded_train_eval[i] for i in idx_evaluation]
data_encoded_train = [data_encoded_train_eval[i] for i in idx_train_eval_set]


logging.info("data_encoded_test rows: {}".format(len(data_encoded_test)))
logging.info("data_encoded_eval rows: {}".format(len(data_encoded_eval)))
logging.info("data_encoded_train rows: {}".format(len(data_encoded_train)))

data_encoded_test


logging.info("Starting data tabulations")


# f = h5py.File("array_in_disk.hdf5", "a", driver="core", backing_store=False)
f = h5py.File("./data/array_in_disk.hdf5", "a")

data_tabulation_step = 20000
end_position = len(data_encoded_train)
rounds = math.ceil(end_position / data_tabulation_step)

for i in range(rounds):
    ini_ = i * data_tabulation_step
    end_ = min((i + 1) * data_tabulation_step, end_position)
    X, Y = construct_x_y_h5py(
        data_to_tabulate=data_encoded_train[ini_:end_],
        remem_window=REMEMBER_WINDOW,
        moving_window=MOVING_WINDOW,
        vocabulary_size=TOP_POPULAR_PRODUCTS,
    )
    if i == 0:
        dset_X_train = f.create_dataset(
            "dataset_X_train", data=X, maxshape=(None, MOVING_WINDOW, TOP_POPULAR_PRODUCTS)
        )

        dset_Y_train = f.create_dataset(
            "dataset_Y_train", data=Y, maxshape=(None, TOP_POPULAR_PRODUCTS)
        )

    else:
        new_rows = X.shape[0]

        dset_X_train.resize(dset_X_train.shape[0] + new_rows, axis=0)
        dset_X_train[-new_rows:] = X

        dset_Y_train.resize(dset_Y_train.shape[0] + new_rows, axis=0)
        dset_Y_train[-new_rows:] = Y

    del X, Y
logging.info("------- data training: Tabulated")
logging.info("Shape of dset_X_train: {}".format(dset_X_train.shape))


data_tabulation_step = 10000
end_position = len(data_encoded_eval)
rounds = math.ceil(end_position / data_tabulation_step)

for i in range(rounds):

    ini_ = i * data_tabulation_step
    end_ = min((i + 1) * data_tabulation_step, end_position)
    X, Y = construct_x_y_h5py(
        data_to_tabulate=data_encoded_eval[ini_:end_],
        remem_window=REMEMBER_WINDOW,
        moving_window=MOVING_WINDOW,
        vocabulary_size=TOP_POPULAR_PRODUCTS,
    )
    if i == 0:
        dset_X_eval = f.create_dataset(
            "dataset_X_eval", data=X, maxshape=(None, MOVING_WINDOW, TOP_POPULAR_PRODUCTS)
        )

        dset_Y_eval = f.create_dataset(
            "dataset_Y_eval", data=Y, maxshape=(None, TOP_POPULAR_PRODUCTS)
        )

    else:
        new_rows = X.shape[0]

        dset_X_eval.resize(dset_X_eval.shape[0] + new_rows, axis=0)
        dset_X_eval[-new_rows:] = X

        dset_Y_eval.resize(dset_Y_eval.shape[0] + new_rows, axis=0)
        dset_Y_eval[-new_rows:] = Y

    del X, Y
logging.info("------- data evaluation: Tabulated")
logging.info("Shape of dset_X_eval: {}".format(dset_X_eval.shape))


data_tabulation_step = 10000
end_position = len(data_encoded_test)
rounds = math.ceil(end_position / data_tabulation_step)

for i in range(rounds):

    ini_ = i * data_tabulation_step
    end_ = min((i + 1) * data_tabulation_step, end_position)
    X, Y = construct_x_y_h5py(
        data_to_tabulate=data_encoded_test[ini_:end_],
        remem_window=REMEMBER_WINDOW,
        moving_window=MOVING_WINDOW,
        vocabulary_size=TOP_POPULAR_PRODUCTS,
    )
    if i == 0:
        dset_X_test = f.create_dataset(
            "dataset_X_test", data=X, maxshape=(None, MOVING_WINDOW, TOP_POPULAR_PRODUCTS)
        )

        dset_Y_test = f.create_dataset(
            "dataset_Y_test", data=Y, maxshape=(None, TOP_POPULAR_PRODUCTS)
        )

    else:
        new_rows = X.shape[0]

        dset_X_test.resize(dset_X_test.shape[0] + new_rows, axis=0)
        dset_X_test[-new_rows:] = X

        dset_Y_test.resize(dset_Y_test.shape[0] + new_rows, axis=0)
        dset_Y_test[-new_rows:] = Y

    del X, Y
logging.info("------- data test: Tabulated")
logging.info("Shape of dset_X_test: {}".format(dset_X_test.shape))


def lazy_property(function):
    attribute = "_" + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


logging.info("Defining Network")


class SequenceClassification:
    def __init__(self, data, target, dropout, num_hidden, num_layers=1, learning_step=0.0001):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.learning_step = learning_step
        self.model_def
        # self.training_
        self.prediction_
        self.error
        self.optimize

    @lazy_property
    def model_def(self):

        network = tf.contrib.rnn.GRUCell(self._num_hidden)
        network = tf.contrib.rnn.DropoutWrapper(network, output_keep_prob=self.dropout)
        # network = tf.contrib.rnn.MultiRNNCell([network] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)

        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        # Softmax layer.
        weight, bias = self._weight_and_bias(self._num_hidden, int(self.target.get_shape()[1]))
        return last, weight, bias

    @lazy_property
    def prediction_(self, training_bool=True):
        # Softmax layer.
        last, weight, bias = self.model_def
        output = tf.nn.softmax(tf.matmul(last, weight) + bias)

        return output

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction_))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = self.learning_step
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.01)

        gradient_vectors = optimizer.compute_gradients(self.cost)
        capped_gradient_vectors = [
            (tf.clip_by_value(grad, -4.0, 4.0), var) for grad, var in gradient_vectors
        ]
        # train_op = optimizer.apply_gradients(capped_gradient_vectors)
        # return optimizer.minimize(self.cost)

        return optimizer.apply_gradients(capped_gradient_vectors)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction_, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


# list what CPUs and GPUs are avalaible
logging.info("Available processing units in machine: {}".format(get_available_processing_units()))


# with tf.device("/gpu:0"):
with tf.device("/cpu:0"):
    data = tf.placeholder(tf.float32, [None, MOVING_WINDOW, TOP_POPULAR_PRODUCTS])
    target = tf.placeholder(tf.float32, [None, TOP_POPULAR_PRODUCTS])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(
        data, target, dropout, num_hidden=NUM_UNITS, learning_step=0.0001
    )


total_batches = math.floor(tmp_value / BATCH_SIZE)
logging.info("Total batches: {}".format(total_batches))
total_eval_records = dset_X_eval.shape[0]


dset_X_train
dset_Y_train
dset_X_train[:, 1].shape

logging.info("Starting TensorFlow training session at {}".format(datetime.now()))
t0 = time.time()  # log time of training start

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# b_num = 0
for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    for batch_number in range(total_batches):

        ini = batch_number * BATCH_SIZE
        end = (batch_number + 1) * BATCH_SIZE
        batch_X = dset_X_train[ini:end]
        batch_Y = dset_Y_train[ini:end]

        sess.run(model.optimize, {data: batch_X, target: batch_Y, dropout: 0.5})

        output = "%d/%d batches processed" % (batch_number + 1, total_batches)
        Printer(output)

    # --------------------------------------------------------------------------------------------------
    #  average error in batches over the evaluation set (measured in error)

    # the evaluation is done selecting random samples from the evaluation data set
    idx_evaluation_ini = np.sort(
        np.random.choice(
            (total_eval_records - batches_eval_size), batches_evaluation, replace=False
        )
    )

    output_error = 0
    output_map_metric = 0

    for batch_number_eval in idx_evaluation_ini:
        ini_e = batch_number_eval
        end_e = batch_number_eval + batches_eval_size

        inputs_ = dset_X_eval[ini_e:end_e]
        actuals_ = dset_Y_eval[ini_e:end_e]

        # classification error
        error = sess.run(
            model.error,
            {data: dset_X_eval[ini_e:end_e], target: dset_Y_eval[ini_e:end_e], dropout: 1},
        )
        output_error = output_error + error

        # MAP@n error
        actuals_decoded = np.apply_along_axis(extract_product, axis=1, arr=actuals_)
        prediction_ = sess.run(model.prediction_, {data: inputs_, dropout: 1})

        prod_recom = np.apply_along_axis(extract_top_n_products, axis=1, arr=prediction_, npos=5)

        to_eval = list(zip(actuals_decoded, prod_recom))
        map_k_vals = list(map(mean_avg_precision_k, to_eval))
        average_metric_batch = reduce(lambda x, y: x + y, map_k_vals) / len(map_k_vals)
        output_map_metric = output_map_metric + average_metric_batch

    time_batch = time.time() - t0
    print(
        " ---> Epoch{:2d} | error eval {:3.1f}% with MAP@5 = {:.6} |{:4.2f}mins".format(
            epoch + 1,
            100 * (output_error / batches_evaluation),
            (output_map_metric / batches_evaluation),
            (time_batch / 60),
        )
    )

logging.info(
    "Finished training at {} taking {} minutes".format(datetime.now(), ((time.time() - t0) / 60))
)


# saver.save(sess, 'my-model-test')
# total evaluation (over the TEST dataset) set MAP@4


number_of_chunks_eval = math.ceil(dset_X_test.shape[0] / prediction_batch)

ini_idx = 0
metric_per_batch = []
products_recom_chunks = []
for i in range(number_of_chunks_eval - 1):

    end_idx = ini_idx + prediction_batch
    actuals_ = dset_Y_test[ini_idx:end_idx]
    inputs_ = dset_X_test[ini_idx:end_idx]

    test_actuals_decoded = np.apply_along_axis(extract_product, axis=1, arr=actuals_)
    prediction_ = sess.run(model.prediction_, {data: inputs_, dropout: 1})
    prod_recom = np.apply_along_axis(extract_top_n_products, axis=1, arr=prediction_, npos=5)
    products_recom_chunks.append(prod_recom)

    to_eval = list(zip(test_actuals_decoded, prod_recom))
    map_k_vals = list(map(mean_avg_precision_k, to_eval))

    average_metric_batch = reduce(lambda x, y: x + y, map_k_vals) / len(map_k_vals)
    metric_per_batch.append(average_metric_batch)
    ini_idx = ini_idx + prediction_batch


map_at_n = reduce(lambda x, y: x + y, metric_per_batch) / len(metric_per_batch)
logging.info("Test set MAP@5: {:.6}".format(map_at_n))
logging.info(
    "Test set Accuracy@1: {:.6}".format(
        accuracy_score(test_actuals_decoded, prod_recom[:, 1]) * 100
    )
)

####################################################################################################
# TEST SET PREDICTIONS
####################################################################################################

products_recommended = np.vstack(products_recom_chunks)
list_suggested_recomnd = list(np.apply_along_axis(join_integers, axis=1, arr=products_recommended))

list_suggested_recomnd

recom_product_coverage = len(np.unique(products_recommended)) / TOP_POPULAR_PRODUCTS
logging.info(
    "Coverage of recommended products (vs TOP_POPULAR_PRODUCTS): {}".format(recom_product_coverage)
)

####################################################################################################
# CLEAN UP SESSION
####################################################################################################

f.close()  # stop writing in h5py file
sess.close()  # stop tensorflow session
