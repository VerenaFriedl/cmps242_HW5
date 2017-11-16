#!/usr/bin/env python2.7

##########
# File: hw5.py
# Authors: Verena Friedl, Jon Akutagawa, Andrew Bailey
# History: created       Nov 9, 2017
#          last changed  Nov 14, 2017
##########

"""

"""

##########
# Imports
##########
from __future__ import print_function
import sys
import os
import re
import numpy as np
import math
import random
import collections
import nltk
import csv
from collections import defaultdict
import splitter

from nltk.corpus import stopwords
#nltk.download('punkt')
from nltk.tokenize.casual import TweetTokenizer

#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.contrib import rnn

#pip install pyenchant
#pip install splitter


##########
# Functions and Classes
##########
############################
# https://gist.github.com/tokestermw/cb87a97113da12acb388
FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result


def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def tokenize(text):

    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@", "<user> ")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smile> ")
    text = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " <neutralface> ")
    text = re_sub(r"<3"," <heart> ")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ")
    text = re_sub(r"#", " <hashtag> ")
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat> ")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong> ")


    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    re_sub(r"([A-Z]{2,})", "<allcaps> " + r"\1")

    text = text.lower()
    # Take care of contractions like don't, can't, it's, ...
    # Glove word model seems to have them as dont, cant, its, ... so just delete the '
    text = re_sub("\'s", "")
    text = re_sub("\'", "")
    text = re_sub("\xe2\x80\x99", "")
    text = re_sub("\xe2\x80\x98", "")

    # special cases
    text = re_sub("shewon", "she won")
    text = re_sub("trumpwon", "trump won")
    text = re_sub("americafirst", "america first")
    text = re_sub("debatenight", "debate night")
    text = re_sub("imwithher", "i am with her")
    text = re_sub("lovetrumpshate", "love trumps hate")
    text = re_sub("trumptrain", "trump train")
    text = re_sub("makeamericagreatagain", "make america great again")
    text = re_sub("americafirst", "america first")
    text = re_sub("thedonald", "the donald")
    text = re_sub("imwithyou", "i am with you")
    text = re_sub("hillarys", "hillary")
    text = re_sub("<user>s", "<user>")
    text = re_sub("\xe2\x80\x94", " - ")
    text = re_sub("\xe2\x80\xa6", "...")
    text = re_sub("\xe2\x80\x9c", " \" ")
    text = re_sub("&amp", " & ")
    text = re_sub("\xe2\x80\x9d", " \" ")
    text = re_sub("&gt", ">")
    text = re_sub("hillaryclinton", "hillary clinton")
    text = re_sub("\xe2\x80\x93", " - ")
    text = re_sub("\xe2\x9c\x93", "")
    text = re_sub("johnkasich", "john kasich")
    text = re_sub("\xe2\x9c\x85", " ")
    text = re_sub("\xc2\xa1", " ")
    text = re_sub("\xe2\x9c\x94\xef\xb8\x8f", " ")
    text = re_sub("realdonaldtrump", "real donald trump")
    # surround special characters with whitespace so they won't be connected to a word
    text = re_sub(r'([\"\,\#\!\:\.\-\?\(\)\;\*\[\]\=\$])', r" \1 ")

    # split text at white spaces
    text = text.split()
    #text = [w for w in text if w.strip()]  # gets rid of strings that are only whitespace(s)
    # -> don't need it if splitting at white space
    return text

############################


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
    labels = [[int(x == 'HillaryClinton'), int(x == 'realDonaldTrump')] for x in labels]
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


def create_glove_dict(glove_path):
    """Create dictionary of words from twitter glove (Global Vectors for Word Representation) data"""
    assert os.path.isfile(glove_path)
    glove = defaultdict(list)
    with open(glove_path, 'r') as glove_f:
        for line in glove_f:
            line = line.split()
            glove[line[0]] = [float(x) for x in line[1:]]
    return dict(glove)


def word_2_vec(sentences, glove, len_glove=50):
    """Covert words to vectors from twitter glove (Global Vectors for Word Representation) data"""
    out_data = []
    sequence_length = []
    errors = []
    for sentence in sentences:
        # sentence = wordninja.split("asd thefactis")

        # words = TweetTokenizer().tokenize(sentence)

        words = tokenize(sentence)
        # words = nltk.word_tokenize(sentence)
        data = []
        # print(words)
        for word in words:
            try:
                data.append(glove[word.lower()])
            # ToDo maybe look into this again and check what errors
            except KeyError as e:
                # print("ERROR", word.lower())
                # errors.append(word.lower())
                # pass
                # Errors for words that are not contained in the word model. Print and take a look at them:
                try:
                    split = tokenize(word)
                    # print("FIRST ERROR", word, split)
                    for word in split:
                        word = splitter.split(word)
                        for sub_word in word:
                            data.append(glove[sub_word.lower()])
                except KeyError as e:
                    # Errors for words that are not contained in the word model. Print and take a look at them:
                    print("ERROR", sub_word.lower())
                    errors.append(sub_word.lower())
                    pass
        out_data.append(np.asarray(data))
        sequence_length.append(len(data))
    print(len(errors))
    max_len_tweet = max([len(tweet) for tweet in out_data])
    padded_output = []
    for data in out_data:
        pad = max_len_tweet - len(data)
        padded = np.concatenate((data, np.zeros([pad, len_glove])))
        padded_output.append(padded)
    return np.asarray(padded_output), np.asarray(sequence_length)


##########
# Main
##########

# Set this to your local path.
#local_path_to_nltk_stopwords = "/Users/vfriedl/Google Drive/classes/cmps242/nltk_data"

# Working directory
work_dir = "/Users/andrewbailey/git/cmps242_HW5/"

# Global index for clinton and trump
clinton_index = 0
trump_index = 1

# Read in data
handles, tweets = readTweetData("./train.csv")
handles_test, tweets_test = readTweetData("./test.csv")

# with open("/Users/andrewbailey/git/cmps242_HW5/train_tweets", 'w+') as t_f:
#     for sentence in tweets:
#         t_f.write(sentence+'\n')
# Tokenize training and test data
# X, train_vectorizer = tokenizeTrainAndTest(tweets, local_path_to_nltk_stopwords)
# X_test = train_vectorizer.transform(tweets_test).toarray()

# (Global Vectors for Word Representation)
input_vector_len = 25  # depends on glove dataset
# place to get glove data https://nlp.stanford.edu/projects/glove/
path_to_glove_twitter = "/Users/andrewbailey/git/cmps242_HW5/glove.twitter.27B."+str(input_vector_len)+"d.txt"#"/home/ubuntu/cmps242_HW5/glove.twitter.27B.50d.txt"
glove = create_glove_dict(path_to_glove_twitter)
print("Glove dict length: "+str(len(glove)))

X, X_seq_len = word_2_vec(tweets, glove, len_glove=input_vector_len)
print("X shape "+str(X.shape))
#print(X[0][0])
#print(X[0][310])
print("X_seq_len shape "+str(X_seq_len.shape))
print("max(X_seq_len) "+str(max(X_seq_len)))
print(X_seq_len)
X_test, test_seq_length = word_2_vec(tweets_test, glove, len_glove=input_vector_len)

# Encode tweet handles in two binary attributes
Y = encodeLabels(handles)
print("Y.shape", Y.shape)
print(Y)

label_vector_len = Y.shape[1]  # 2

# create validation dataset
# ToDo we should randomize this but it works for now
len_val_set = 1000
X_train = X[len_val_set:]
X_val = X[:len_val_set]
Y_train = Y[len_val_set:]
Y_val = Y[:len_val_set]

print("X_train.shape ", X_train.shape)
print("X_val.shape ", X_val.shape)
print("Y_train.shape ", Y_train.shape)
print("Y_val.shape ", Y_val.shape)

train_seq_length = X_seq_len[len_val_set:]
val_seq_length = X_seq_len[:len_val_set]

print("train_seq_length ", train_seq_length.shape)
print("val_seq_length ", val_seq_length.shape)


# expected data input format
training_labels = collections.namedtuple('training_data', ['input', 'seq_len', 'label'])
inference_labels = collections.namedtuple('inference_data', ['input', 'seq_len'])

train_data = training_labels(input=X_train, seq_len=train_seq_length, label=Y_train)
#print("train_data "+str(train_data))
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
prefetch_buffer_size = 50
# number of iterations through all training data
n_epochs = 10000
# number of hidden nodes in blstm
n_hidden = [200]
forget_bias = 5
learning_rate = 0.001
#######################
# set output dir for saving model
model_name = "3lstm_fnn"
output_dir = work_dir+"logs/"+model_name #"/home/ubuntu/cmps242_HW5/logs/3lstm"

# path to trained model and model dir
# trained_model = tf.train.latest_checkpoint("/home/ubuntu/cmps242_HW5/logs")
# print("trained model path", trained_model)
trained_model_dir = work_dir+"logs" #"/home/ubuntu/cmps242_HW5/logs"
training_iters = 2 #10000
trained_model = output_dir+"/"+model_name+"-"+str(training_iters) #"/Users/andrewbailey/git/cmps242_HW5/logs/2blstm_fnn-10"

########
training = True
# goes really fast when batch size is high
batch_size = 250
output_csv = work_dir+"test_submission.csv" #"/home/ubuntu/cmps242_HW5/test_submission.csv"


# there is way easier to use a feed_dict for the sess.run section at the bottom but I had this code so I implemented it
def create_dataset():
    """Create placeholders for data and iterator"""
    # create placeholders for the values we are going to feed into the network
    place_X = tf.placeholder(tf.float32, shape=[None, None, input_vector_len], name='Input')
    place_Seq = tf.placeholder(tf.int32, shape=[None], name='Sequence_Length')
    place_Y = tf.placeholder(tf.int32, shape=[None, label_vector_len], name='Label')
    # create dataset objects which will allow us to use the Dataset class from tensorflow which deals with
    # the queue operations and the flow of data into the graph
    datasetX = tf.data.Dataset.from_tensor_slices(place_X)
    datasetSeq = tf.data.Dataset.from_tensor_slices(place_Seq)
    datasetY = tf.data.Dataset.from_tensor_slices(place_Y)

    if training:
        dataset = tf.data.Dataset.zip((datasetX, datasetSeq, datasetY))
        dataset = dataset.shuffle(buffer_size=1000)
        # dataset = dataset.padded_batch(batch_size, dataset.output_shapes)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(n_epochs)
    else:
        dataset = tf.data.Dataset.zip((datasetX, datasetSeq))
        dataset = dataset.batch(batch_size)
        # dataset = dataset.padded_batch(batch_size, dataset.output_shapes)

    # prefetch data to make computation quicker so the graph doesnt have to wait for data to be transferred
    dataset = dataset.prefetch(buffer_size=5)

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
    x_shape = x.get_shape().as_list()
    print("x_shape in create_graph "+str(x_shape))
    # input = tf.reshape(x, shape=[-1, x_shape, 1])
    with tf.variable_scope("LSTM_layers"):
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size, forget_bias=forget_bias) for size in n_hidden]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        # 'outputs' is a tensor of shape [batch_size, max_time, 256]
        # 'state' is a N-tuple where N is the number of LSTMCells containing a
        # tf.contrib.rnn.LSTMStateTuple for each cell
        output, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                      inputs=x,
                                      dtype=tf.float32,
                                      time_major=False,
                                      sequence_length=seq_len)

    # with tf.variable_scope("BLSTM_layer1"):
    #     # Forward direction cell
    #     lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
    #     # Backward direction cell
    #     lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=forget_bias, state_is_tuple=True)
    #
    #     # returns output nodes and the hidden states
    #     outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
    #                                                  lstm_bw_cell, input,
    #                                                  dtype=tf.float32,
    #                                                  sequence_length=seq_len)
    #     # concat two output layers so we can treat as single output layer
    #     output = tf.concat(outputs, 2)


    # Not sure if this function works properly. Found other code from the tutorial in the lecture to replace it
    # Would not use this anymore
    def last_relevant(output, length):
        """Collect last relevant output from a batch of samples
        https://danijar.com/variable-sequence-lengths-in-tensorflow/
        """
        batch_size = tf.shape(output)[0]
        print("batch_size "+str(batch_size))
        max_length = tf.shape(output)[1]
        print("max_length " + str(max_length))
        out_size = int(output.get_shape()[2])
        print("out_size " + str(out_size))
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        print("flat:"+str(flat)+" index:"+str(index))
        relevant = tf.gather(flat, index)
        #print("relevant: "+str(relevant))
        return relevant

    # Using the last_relevant function, that might have a bug. Replaced by code below.
    #last_outputs = last_relevant(output, seq_len)

    # Replacing the function last_relevant
    # found in the tutorial presented in the lecture.
    batch_size = tf.shape(output)[0]
    last_outputs = tf.gather_nd(output, tf.stack([tf.range(batch_size), seq_len - 1], axis=1))


    # create fully connected input layer
    with tf.variable_scope("Full_conn_layer1"):
        input_dim = int(last_outputs.get_shape()[1])
        weights = tf.get_variable(name="weights", shape=[input_dim, label_vector_len],
                                  initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(2.0 / (2 * label_vector_len))))
        bias = tf.get_variable(name="bias", shape=[label_vector_len],
                               initializer=tf.zeros_initializer)

        # could put activation function here but the loss function completes this for us
        final_output = tf.nn.bias_add(tf.matmul(last_outputs, weights), bias)
    return final_output


# initializing a step variable
global_step = tf.Variable(0, name='global_step', trainable=False)
# if computer has nvidia-smi gpu
GPU = False

if training:
    # variable scope is to allow reuse of weights between validation graph and training graph
    if GPU:
        device = '/gpu:0'
    else:
        device = '/cpu:0'

    with tf.variable_scope("graph", reuse=tf.AUTO_REUSE) and tf.device(device):
        train_output = create_graph(x_train, seq_len_train)
        print("train_output.get_shape() "+str(train_output.get_shape()))
        # create get index of each argument
        y_label_indices = tf.argmax(y_train, 1, name="train_label_indices")
        # loss function
        train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_output,
                                                                    labels=y_label_indices)
        # minimize average loss over entire batch
        train_cost = tf.reduce_mean(train_loss)
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
        print("val_output.get_shape() " + str(val_output.get_shape()))
        # loss function
        y_label_indices_val = tf.argmax(y_val, 1, name="val_label_indices")
        val_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=val_output,
                                                                  labels=y_label_indices_val)
        # minimize average loss over entire batch
        val_cost = tf.reduce_mean(val_loss)
        val_variable_summaries(val_cost)
        # create prediction by comparing index from output and label then getting the mean
        val_predict = tf.equal(tf.argmax(val_output, 1), tf.argmax(y_val, 1))
        val_accuracy = tf.reduce_mean(tf.cast(val_predict, tf.float32))
        # add accuracy to variable summary so we can track using tensorboard
        val_variable_summaries(val_accuracy)

    val_summary = tf.summary.merge(validation_summaries)
    train_summary = tf.summary.merge(training_summaries)
else:
    test_output = create_graph(x_test, seq_len_test)
    print("test_output.get_shape()"+str(test_output.get_shape()))
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
        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)
        saver.save(sess, save_model_path,
                   global_step=global_step)

    else:
        # if we want to load the model
        writer = tf.summary.FileWriter(trained_model_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
        saver.restore(sess, trained_model)
        # TODO could have a bug here if using wrong config file with wrong model name
        save_model_path = os.path.join(trained_model, model_name)

    # load data into placeholders
    if training:
        # print(type(train_data.input))
        #print(type(train_data.label))
        #print(type(train_data.seq_len))
        #print(type(place_X_train))
        #print(type(place_Y_train))
        #print(type(place_Seq_train))

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
                #if step % 10 == 0:
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
            spamwriter.writerows([[i, x[trump_index], x[clinton_index]] for i, x in enumerate(all_data_list)])
# close session and writer
sess.close()
writer.close()
