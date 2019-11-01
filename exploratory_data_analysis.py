import numpy as np
import pandas as pd
import time
import datetime
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

import mlflow
from ml_metrics import average_precision
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.set_palette(sns.color_palette("viridis"))
sns.set_context("notebook")
warnings.simplefilter(action="ignore", category=FutureWarning)

####################################################################################################
# 🚀 LOAD DATA
####################################################################################################

WEEKS_OF_DATA = 3  # load 1,2 or 3 weeks of data (current implementation is 1)

DATA_PATH1 = "./data/ga_product_sequence_20191013.csv"
DATA_PATH2 = "./data/ga_product_sequence_20191020.csv"
DATA_PATH3 = "./data/ga_product_sequence_20191027.csv"
INPUT_VAR = "product_sequence"

# constants
N_TOP_PRODUCTS = 10000  # 6000 is ~70% views, 8000 ~80%, 10000 ~84%, 12000 ~87%, 15000 ~90%


####################################################################################################
# 🚀 PREPARE SEQUENCE DATA
####################################################################################################
print("     Reading raw input data")

if WEEKS_OF_DATA == 1:
    sequence_df = pd.read_csv(DATA_PATH1)
elif WEEKS_OF_DATA == 2:
    sequence_df = pd.read_csv(DATA_PATH1)
    sequence_df2 = pd.read_csv(DATA_PATH2)
    sequence_df = sequence_df.append(sequence_df2)
elif WEEKS_OF_DATA == 3:
    sequence_df = pd.read_csv(DATA_PATH1)
    sequence_df2 = pd.read_csv(DATA_PATH2)
    sequence_df3 = pd.read_csv(DATA_PATH3)
    sequence_df = sequence_df.append(sequence_df2).append(sequence_df3)
    del sequence_df2
else:
    sequence_df = pd.read_csv(DATA_PATH1)

MIN_DATE, MAX_DATE = sequence_df["visit_date"].min(), sequence_df["visit_date"].max()

print("     Data contains {} sequences from {} to {}".format(len(sequence_df), MIN_DATE, MAX_DATE))
print("     Including top {} most popular products".format(N_TOP_PRODUCTS))

print("     Tokenizing, padding, filtering & splitting sequences")
# define tokenizer to encode sequences while including N most popular items (occurence)
tokenizer = keras.preprocessing.text.Tokenizer(num_words=N_TOP_PRODUCTS)
# encode sequences
tokenizer.fit_on_texts(sequence_df["product_sequence"])
sequences = tokenizer.texts_to_sequences(sequence_df["product_sequence"])

# get information on session lengths
sequence_lengths = [(len(session)) for session in sequences]

sequence_lengths.count(2) / len(sequence_lengths)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
sns.countplot(sequence_lengths)
ax.set(xlabel='Sequence Length', ylabel='Count', title='Distribution of Sequence Lengths')
ax.set_xlim(right=10.5)
plt.tight_layout()
