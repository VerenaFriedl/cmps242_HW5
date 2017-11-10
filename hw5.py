#!/usr/bin/env python3

##########
# File: hw5.py
# Authors: Verena Friedl, Jon Akutagawa, Andrew Bailey
# History: created       Nov 9, 2017
#          last changed  Nov 9, 2017
##########

"""

"""

##########
# Imports
##########
from __future__ import print_function
import sys
import os
import numpy as np
import math
import random
import collections
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.contrib import rnn


##########
# Functions and Classes
##########


def readTweetData(filename):
    """
    Read in tweet collection of "HillaryClinton" and "realDonaldTrump" (or "none in case of test data). 
    Other twitter handles would break this function.
    :param filename: File to read tweet data from.
    :return: list with handles and list with tweets; in order of file appearance.
    """
    fileH = open(filename, 'r')
    header = fileH.readline().rstrip().split(',')
    handles = []
    tweets = []
    goodHandles = ["HillaryClinton", "realDonaldTrump", "none"]
    for line in fileH.readlines():
        splitline = line.rstrip('\n').split(',')

        # line starts with twitter handle
        if splitline[0] in goodHandles:
            handles.append(splitline[0])
            tweets.append(','.join(splitline[1:]))
        # line is part of the last tweet
        else:
            tweets[len(tweets) - 1] = tweets[len(tweets) - 1] + "\n" + ','.join(splitline)

    # write to file to test if data is read in correctly (should be exactly the same as the input file)
    # outfileH = open('./out.csv','w')
    # outfileH.write(",".join(header) + "\n")
    # for i in range(0, len(handles)):
    #   outfileH.write(handles[i] + "," + tweets[i]+"\n")

    return handles, tweets


def encodeLabels(list):
    """
    Encode labels in 0 and 1 for comparing "HillaryClinton" = 1 to "realDonaldTrump" = 0.
    :param list: input labels
    :return: numpy array with 0 and 1
    """
    labels = list
    labels = [[int(x == 'HillaryClinton'), int(x != 'HillaryClinton')] for x in labels]
    labels = np.asarray(labels)
    return labels


def tokenizeTrainAndTest(trainData, testData, stopword_path):
    """
    Tokenize the tweets.
    :param trainData: list of training tweets.
    :param testData: list of test tweets.
    :return: tokenized training data, tokenized test data.
    """
    # local path for the downloaded nltk data
    nltk.data.path.append(stopword_path)
    vectorizer = TfidfVectorizer(input='content', stop_words=stopwords.words('english'), decode_error='ignore',
                                 norm='l2')
    # merge train and test message lists for encoding
    trainAndTest = trainData + testData
    X_trainAndTest = vectorizer.fit_transform(trainAndTest).toarray()
    # split encoded train and test data
    X = X_trainAndTest[:len(tweets)]
    X_test = X_trainAndTest[len(tweets):]
    return X, X_test


##########
# Main
##########

# Set this to your local path.
local_path_to_nltk_stopwords = "/Users/vfriedl/Google Drive/classes/cmps242/nltk_data"

# Global index for clinton and trump
clinton_index = 0
trump_index = 1

# Read in data
handles, tweets = readTweetData("./train.csv")
handles_test, tweets_test = readTweetData("./test.csv")

# Encode tweet handles in two binary attributes
Y = encodeLabels(handles)
# labels_trump = np.asarray([abs(x-1) for x in labels_hillary])
# Y = np.matrix([labels_hillary, labels_trump]).T
print("Y.shape", Y.shape)

# Tokenize training and test data
X, X_test = tokenizeTrainAndTest(tweets, tweets_test, local_path_to_nltk_stopwords)
print("X.shape", X.shape)
# print(X_test.shape)
input_vector_len = X.shape[1]
label_vector_len = Y.shape[1]
train_seq_length = [input_vector_len for x in range(X.shape[0])]
test_seq_length = [input_vector_len for x in range(X_test.shape[0])]

# expected data input format
training_labels = collections.namedtuple('training_data', ['input', 'seq_len', 'label'])
inference_labels = collections.namedtuple('inference_data', ['input', 'seq_len'])

train_data = training_labels(input=X, seq_len=train_seq_length, label=Y)
test_data = inference_labels(input=X_test, seq_len=test_seq_length)
########
# hyperparameters
batch_size = 5
training = True
shuffle_buffer_size = 50
prefetch_buffer_size = 5
# number of iterations through all training data
n_epochs = 10
# number of hidden nodes in blstm
n_hidden = 10
forget_bias = 5
learning_rate = 0.001
########
# create placeholders for the values we are going to feed into the network
place_X = tf.placeholder(tf.float32, shape=[None, input_vector_len], name='Input')
place_Seq = tf.placeholder(tf.int32, shape=[None], name='Sequence_Length')
place_Y = tf.placeholder(tf.int32, shape=[None, label_vector_len], name='Label')

# create dataset objects which will allow us to use the Dataset class from tensorflow which deals with
# the queue operations and the flow of data into the graph
datasetX = tf.data.Dataset.from_tensor_slices(place_X)
datasetSeq = tf.data.Dataset.from_tensor_slices(place_Seq)
datasetY = tf.data.Dataset.from_tensor_slices(place_Y)

# batch the data into the batch size we want
batchX = datasetX.batch(batch_size)
batchSeq = datasetSeq.batch(batch_size)
batchY = datasetY.batch(batch_size)

if training:
    dataset = tf.data.Dataset.zip((batchX, batchSeq, batchY))
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
else:
    dataset = tf.data.Dataset.zip((batchX, batchSeq))

# prefetch data to make computation quicker so the graph doesnt have to wait for data to be transferred
dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

# create iterator which will iterate through dataset
iterator = dataset.make_initializable_iterator()

if training:
    x, seq_len, y = iterator.get_next()
else:
    x, seq_len = iterator.get_next()

# reshape for blstm layer
input = tf.reshape(x, shape=[-1, input_vector_len, 1])
print(input.get_shape())

with tf.variable_scope("BLSTM_layer1"):
    # Forward direction cell
    lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
    # Backward direction cell
    lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)

    # returns output nodes and the hidden states
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                 lstm_bw_cell, input,
                                                 dtype=tf.float32,
                                                 sequence_length=seq_len)
    # concat two output layers so we can treat as single output layer
    output = tf.concat(outputs, 2)

with tf.variable_scope("BLSTM_layer2"):
    # Forward direction cell
    lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
    # Backward direction cell
    lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
    # returns output nodes and the hidden states
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, output,
                                                 dtype=tf.float32,
                                                 sequence_length=seq_len)
    # concat two output layers so we can treat as single output layer
    output = tf.concat(outputs, 2)

print(output.get_shape())
output_reshape_2 = tf.reshape(output, [-1, 2 * n_hidden * input_vector_len])
print(output_reshape_2.get_shape())

# create fully connected input layer
with tf.variable_scope("Full_conn_layer1"):
    input_dim = int(output_reshape_2.get_shape()[1])
    weights = tf.get_variable(name="weights", shape=[input_dim, label_vector_len],
                              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / (2 * label_vector_len))))
    bias = tf.get_variable(name="bias", shape=[label_vector_len],
                           initializer=tf.zeros_initializer)

    # could put activation function here but the loss function completes this for us
    final_output = tf.nn.bias_add(tf.matmul(output_reshape_2, weights), bias)

print(final_output.get_shape())

# create get index of each argument
y_label_indices = tf.argmax(y, 1, name="y_label_indices")

# loss function
loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_output,
                                                       labels=y_label_indices)
# minimize average loss over entire batch
cost = tf.reduce_mean(loss1)

predict = tf.equal(tf.argmax(final_output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
global_step = tf.Variable(0, name='global_step', trainable=False)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

#######################
# graph has been completed
#######################

# set output dir for saving model
output_dir = "/Users/andrewbailey/git/cmps242_HW5/logs"
model_name = "2blstm_fnn"
# path to trained model and model dir
trained_model = "some/path"
trained_model_dir = "another/path"

training_iters = 1000

########
# create config for training session
config = tf.ConfigProto(log_device_placement=False,
                        intra_op_parallelism_threads=8,
                        allow_soft_placement=True)
# shows gpu memory usage
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # create logs
    if training:
        # initialize
        writer = tf.summary.FileWriter(output_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        save_model_path = os.path.join(output_dir, model_name)
        saver = tf.train.Saver()
        saver.save(sess, save_model_path,
                   global_step=global_step)
        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)
    else:
        # if we want to load the model
        writer = tf.summary.FileWriter(trained_model_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
        saver.restore(sess, trained_model)
        # TODO could have a bug here if using wrong config file with wrong model name
        save_model_path = os.path.join(trained_model, model_name)

    # load data into placeholders
    if training:
        sess.run(iterator.initializer,
                 feed_dict={place_X: train_data.input,
                            place_Y: train_data.label,
                            place_Seq: train_data.seq_len})
    else:
        sess.run(iterator.initializer,
                 feed_dict={place_X: train_data.input,
                            place_Seq: train_data.seq_len})

    # Keep training until reach max iterations
    print("Training Has Started!", file=sys.stderr)
    step = 0
    # we only want to write the meta graph once cause it is so big
    write_once = True
    try:
        while step < training_iters:
            # Run optimization training step
            _ = sess.run([train_op])
            step += 1
            # get training accuracy stats
            if step % 10 == 0:
                global_step1, train_acc, train_cost = sess.run([global_step,
                                                               accuracy,
                                                               cost])
                # add summary statistics
                # writer.add_summary(summary, global_step)
                # print summary stats
                print("Iter " + str(global_step1) + ", Training Cost= " +
                      "{:.6f}".format(train_cost) + ", Training Accuracy= " +
                      "{:.5f}".format(train_acc), file=sys.stderr)
                # if it has been enough time save model and print training stats

                saver.save(sess, save_model_path,
                           global_step=global_step, write_meta_graph=write_once)
                write_once = False
    # catches errors if dataset is finished all epochs
    # increase epochs if we have not converged
    except tf.errors.OutOfRangeError:
        print("End of dataset")
# close session and writer
sess.close()
writer.close()
