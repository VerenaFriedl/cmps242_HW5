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
import csv

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
    labels = [[int(x == 'realDonaldTrump')] for x in labels]
    labels = np.asarray(labels)
    return labels


def tokenizeTrainAndTest(trainData, stopword_path):
    """
    Tokenize the tweets.
    :param trainData: list of training tweets.
    :return: tokenized training data, vectorizer.
    """
    # local path for the downloaded nltk data
    nltk.data.path.append(stopword_path)
    vectorizer = TfidfVectorizer(input='content', stop_words=stopwords.words('english'), decode_error='ignore',
                                 norm='l2')
    X = vectorizer.fit_transform(trainData).toarray()

    # split encoded train and test data
    return X, vectorizer

##########
# Main
##########

# Set this to your local path.
local_path_to_nltk_stopwords = "/Users/vfriedl/Google Drive/classes/cmps242/nltk_data"

# Global index for clinton and trump
# clinton_index = 0
# trump_index = 1
# We are now classifying if a tweet is from realDonaldTrump or not

# Read in data
handles, tweets = readTweetData("./train.csv")
handles_test, tweets_test = readTweetData("./test.csv")

# Tokenize training and test data
X, train_vectorizer = tokenizeTrainAndTest(tweets, local_path_to_nltk_stopwords)
X_test = train_vectorizer.transform(tweets_test).toarray()

# Encode tweet handles in two binary attributes
Y = encodeLabels(handles)
print("Y.shape", Y.shape)

# create validation dataset we should randomize this but it works for now
len_val_set = 1000
X_train = X[len_val_set:]
X_val = X[:len_val_set]
Y_train = Y[len_val_set:]
Y_val = Y[:len_val_set]

print("X.shape", X_train.shape)
input_vector_len = X_train.shape[1]
label_vector_len = Y.shape[1]  # 2
train_seq_length = [input_vector_len for x in range(X_train.shape[0])]
test_seq_length = [input_vector_len for x in range(X_test.shape[0])]
val_seq_length = [input_vector_len for x in range(X_test.shape[0])]

# expected data input format
training_labels = collections.namedtuple('training_data', ['input', 'seq_len', 'label'])
inference_labels = collections.namedtuple('inference_data', ['input', 'seq_len'])

train_data = training_labels(input=X_train, seq_len=train_seq_length, label=Y_train)
val_data = training_labels(input=X_val, seq_len=val_seq_length, label=Y_val)
test_data = inference_labels(input=X_test, seq_len=test_seq_length)

# keep track of data using tensorboard (tensorboard --logdir logs)
training_summaries = []
validation_summaries = []


def val_variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    source: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
    """
    with tf.name_scope("validation"):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            summary = tf.summary.scalar('mean', mean)
            validation_summaries.append(summary)


def train_variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    source: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
    """
    with tf.name_scope("training"):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            summary = tf.summary.scalar('mean', mean)
            training_summaries.append(summary)


########
# hyperparameters
shuffle_buffer_size = 50
prefetch_buffer_size = 5
# number of iterations through all training data
n_epochs = 10
# number of hidden nodes in blstm
n_hidden = 10
forget_bias = 5
learning_rate = 0.001
#######################
# set output dir for saving model
output_dir = "/Users/andrewbailey/git/cmps242_HW5/logs"
model_name = "2blstm_fnn"
# path to trained model and model dir
trained_model = tf.train.latest_checkpoint("/Users/andrewbailey/git/cmps242_HW5/logs")
print("trained model path", trained_model)
# trained_model = "/Users/andrewbailey/git/cmps242_HW5/logs/2blstm_fnn-10"
trained_model_dir = "/Users/andrewbailey/git/cmps242_HW5/logs"
training_iters = 1000
########
training = True
# goes really fast when batch size is high
batch_size = 2
output_csv = "/Users/andrewbailey/git/cmps242_HW5/test_submission.csv"

# there is way easier to use a feed_dict for the sess.run section at the bottom but I had this code so I implemented it
def create_dataset():
    """Create placeholders for data and iterator"""
    # create placeholders for the values we are going to feed into the network
    place_X = tf.placeholder(tf.float32, shape=[None, input_vector_len], name='Input')
    place_Seq = tf.placeholder(tf.int32, shape=[None], name='Sequence_Length')
    place_Y = tf.placeholder(tf.int32, shape=[None, label_vector_len], name='Label')
    # create dataset objects which will allow us to use the Dataset class from tensorflow which deals with
    # the queue operations and the flow of data into the graph
    datasetX = tf.data.Dataset.from_tensor_slices(place_X)
    datasetSeq = tf.data.Dataset.from_tensor_slices(place_Seq)
    datasetY = tf.data.Dataset.from_tensor_slices(place_Y)

    if training:
        dataset = tf.data.Dataset.zip((datasetX, datasetSeq, datasetY))
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(n_epochs)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    else:
        dataset = tf.data.Dataset.zip((datasetX, datasetSeq))
        # maybe we can do this bud idk how it works at end of dataset
        dataset = dataset.batch(batch_size)

    # prefetch data to make computation quicker so the graph doesnt have to wait for data to be transferred
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    # create iterator which will iterate through dataset
    iterator = dataset.make_initializable_iterator()
    return place_X, place_Seq, place_Y, iterator


# create placeholders
if training:
    with tf.variable_scope("training_data"):
        place_X_train, place_Seq_train, place_Y_train, iterator_train = create_dataset()
        x_train, seq_len_train, y_train = iterator_train.get_next()
    with tf.variable_scope("validation_data"):
        place_X_val, place_Seq_val, place_Y_val, iterator_val = create_dataset()
        x_val, seq_len_val, y_val = iterator_val.get_next()
else:
    with tf.variable_scope("test_data"):
        place_X_test, place_Seq_test, place_Y_test, iterator_test = create_dataset()
        x_test, seq_len_test = iterator_test.get_next()



def create_graph(x, seq_len):
    """Create graph using outputs from iterators"""
    # reshape for blstm layer
    input = tf.reshape(x, shape=[-1, input_vector_len, 1])
    print("input shape", input.get_shape())
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
    return final_output


# initializing a step variable
global_step = tf.Variable(0, name='global_step', trainable=False)
# if computer has nvidia-smi gpu
GPU = False

if training:
    # variable scope is to allow reuse of weights betwen validation graph and training grapb
    if GPU:
        device = '/gpu:0'
    else:
        device = '/cpu:0'

    with tf.variable_scope("graph", reuse=tf.AUTO_REUSE) and tf.device(device):
        train_output = create_graph(x_train, seq_len_train)
        print(train_output.get_shape())
        # create get index of each argument
        y_label_indices = tf.argmax(y_train, 1, name="y_label_indices")
        # loss function
        loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_output,
                                                               labels=y_label_indices)
        # minimize average loss over entire batch
        train_cost = tf.reduce_mean(loss1)
        # add cost to variable summary so we can track using tensorboard
        train_variable_summaries(train_cost)
        # create optimizer
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_cost, global_step=global_step)

        # create prediction by comparing index from output and label then getting the mean
        train_predict = tf.equal(tf.argmax(train_output, 1), tf.argmax(y_train, 1))
        train_accuracy = tf.reduce_mean(tf.cast(train_predict, tf.float32))
        train_variable_summaries(train_accuracy)

    with tf.variable_scope("graph", reuse=tf.AUTO_REUSE):
        # create graph and
        val_output = create_graph(x_val, seq_len_val)
        # loss function
        y_label_indices_val = tf.argmax(y_val, 1, name="y_label_indices")
        val_loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=val_output,
                                                                   labels=y_label_indices_val)
        # minimize average loss over entire batch
        val_cost = tf.reduce_mean(val_loss1)
        train_variable_summaries(val_cost)
        # create prediction by comparing index from output and label then getting the mean
        val_predict = tf.equal(tf.argmax(val_output, 1), tf.argmax(y_val, 1))
        val_accuracy = tf.reduce_mean(tf.cast(val_predict, tf.float32))
        # add accuracy to variable summary so we can track using tensorboard
        val_variable_summaries(val_accuracy)

    val_summary = tf.summary.merge(validation_summaries)
    train_summary = tf.summary.merge(training_summaries)
else:
    test_output = create_graph(x_test, seq_len_test)
    print(test_output.get_shape())
    # create get index of each argument
    test_probs = tf.nn.softmax(test_output)

#######################
# graph has been completed
#######################
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
        sess.run(iterator_train.initializer,
                 feed_dict={place_X_train: train_data.input,
                            place_Y_train: train_data.label,
                            place_Seq_train: train_data.seq_len})
        sess.run(iterator_val.initializer,
                 feed_dict={place_X_val: val_data.input,
                            place_Y_val: val_data.label,
                            place_Seq_val: val_data.seq_len})
    else:
        sess.run(iterator_test.initializer,
                 feed_dict={place_X_test: test_data.input,
                            place_Seq_test: test_data.seq_len})

    if training:
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
                if step % 1 == 0:
                    summary_v, val_acc, val_cost1 = sess.run([val_summary,
                                                             val_accuracy,
                                                             val_cost])
                    summary_t, global_step1, train_acc, train_cost1 = sess.run([train_summary, global_step,
                                                                                 train_accuracy,
                                                                                 train_cost])

                    # add summary statistics
                    writer.add_summary(summary_t, global_step1)
                    writer.add_summary(summary_v, global_step1)

                    # print summary stats
                    print("Iter " + str(global_step1) + ", Training Cost= " +
                          "{:.6f}".format(train_cost1) + ", Validation Cost= " +
                          "{:.5f}".format(val_cost1), file=sys.stderr)
                    # if it has been enough time save model and print training stats
                    print("Iter " + str(global_step1) + ", Training Accuracy= " +
                          "{:.6f}".format(train_acc) + ", Validation Accuracy= " +
                          "{:.5f}".format(val_acc), file=sys.stderr)

                    saver.save(sess, save_model_path,
                               global_step=global_step1, write_meta_graph=write_once)
                    write_once = False
        # catches errors if dataset is finished all epochs
        # increase epochs if we have not converged
        except tf.errors.OutOfRangeError:
            print("End of dataset")
    else:
        all_data_list = []
        try:
            while True:
                prob, xt = sess.run([test_probs, x_test])
                # print(xt == train_vectorizer.transform([tweets_test[i]]).toarray())
                all_data_list.extend(np.vstack(prob))
                print(len(all_data_list))
        except tf.errors.OutOfRangeError:
            print("End of dataset")
        # make sure we got all the data
        assert len(tweets_test) == len(all_data_list)
        # write out file
        with open(output_csv, 'w+') as csv_file:
            spamwriter = csv.writer(csv_file, delimiter=',')  # ,quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["id", "realDonaldTrump", "HillaryClinton"])
            spamwriter.writerows([[i, x, 1-x] for i, x in enumerate(all_data_list)])
# close session and writer
sess.close()
writer.close()



